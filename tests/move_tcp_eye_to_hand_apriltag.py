#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Eye-to-Hand AprilTag TCP到位测试脚本。

流程:
1) 读取 results/eye_to_hand/handeye_transform.txt 得到 T_base_camera
2) 相机每帧先去畸变，再由 pyapriltags 直接估计 T_camera_tag
3) 根据 T_base_tcp_target = T_base_camera @ T_camera_tag @ T_tag_tcp_target 计算目标TCP
4) 按键触发后，使用 ur_rtde 1.6.0 通过 moveJ(getInverseKinematics) 执行运动

按键:
- Enter: 锁定当前目标位姿并立即执行运动
- c: 清除已锁定目标位姿
- Esc: 退出
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from handeye.camera.realsense import RealSenseCamera
from handeye.config import APRILTAG_TEST_CONFIG, UR3_CONFIG


@dataclass
class RuntimeConfig:
    mode: str
    robot_ip: str
    handeye_file: str
    tag_family: str
    tag_size: float
    target_tag_id: int
    decision_margin_threshold: float
    axis_length: float
    t_tag_tcp_target: np.ndarray
    rtde_velocity: float
    rtde_acceleration: float
    dry_run: bool


def _cfg_str(key: str, default: str) -> str:
    val = APRILTAG_TEST_CONFIG.get(key, default)
    return str(val)


def _cfg_float(key: str, default: float) -> float:
    val = APRILTAG_TEST_CONFIG.get(key, default)
    if isinstance(val, bool):
        return float(int(val))
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    if isinstance(val, str):
        return float(val)
    raise TypeError(f"配置项 {key} 不是可转换的数值类型: {type(val)}")


def _cfg_int(key: str, default: int) -> int:
    val = APRILTAG_TEST_CONFIG.get(key, default)
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        return int(val)
    raise TypeError(f"配置项 {key} 不是可转换的整型: {type(val)}")


def _cfg_bool(key: str, default: bool) -> bool:
    val = APRILTAG_TEST_CONFIG.get(key, default)
    return bool(val)


def _create_apriltag_detector(family: str) -> Any:
    try:
        from pyapriltags import Detector
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"pyapriltags 导入失败: {exc}") from exc

    return Detector(
        families=family,
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )


def _create_rtde_interfaces(robot_ip: str) -> tuple[Any, Any]:
    try:
        import rtde_control
        import rtde_receive
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"ur_rtde 导入失败: {exc}") from exc

    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    return rtde_c, rtde_r


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AprilTag TCP位姿到位测试 (支持 eye_to_hand / eye_on_hand)")
    parser.add_argument(
        "--robot-ip",
        type=str,
        default=str(UR3_CONFIG.get("tcp_host_ip", "192.168.56.102")),
        help="UR机器人IP"
    )
    parser.add_argument(
        "--handeye-file",
        type=str,
        default="",
        help="手眼结果矩阵文件路径；为空时根据 mode 自动选择"
    )
    parser.add_argument(
        "--target-tag-id",
        type=int,
        default=_cfg_int("target_tag_id", 0),
        help="目标AprilTag ID"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅计算与显示，不发送机器人运动命令"
    )
    return parser.parse_args()


def _select_mode_interactive() -> str:
    print("\n请选择手眼模式:")
    print("  1) eye_to_hand (眼在手外)")
    print("  2) eye_on_hand (眼在手上)")
    while True:
        choice = input("请输入 1 或 2 (默认1): ").strip()
        if choice == "":
            return "eye_to_hand"
        if choice == "1":
            return "eye_to_hand"
        if choice == "2":
            return "eye_on_hand"
        print("输入无效，请输入 1 或 2")


def _build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    target_pose_any = APRILTAG_TEST_CONFIG.get("t_tag_tcp_target", [0, 0, 0.1, 0, 0, 0])
    if not isinstance(target_pose_any, Sequence) or len(target_pose_any) != 6:
        raise ValueError("APRILTAG_TEST_CONFIG['t_tag_tcp_target'] 必须是6维 [x,y,z,rx,ry,rz]")
    target_pose = np.asarray([float(v) for v in target_pose_any], dtype=np.float64)
    if target_pose.shape != (6,):
        raise ValueError("APRILTAG_TEST_CONFIG['t_tag_tcp_target'] 必须是6维 [x,y,z,rx,ry,rz]")

    mode = _select_mode_interactive()
    default_handeye_file = os.path.join(ROOT, "results", mode, "handeye_transform.txt")
    handeye_file = str(args.handeye_file) if str(args.handeye_file).strip() else default_handeye_file

    cfg = RuntimeConfig(
        mode=mode,
        robot_ip=args.robot_ip,
        handeye_file=handeye_file,
        tag_family=_cfg_str("tag_family", "tag36h11"),
        tag_size=_cfg_float("tag_size", 0.04),
        target_tag_id=int(args.target_tag_id),
        decision_margin_threshold=_cfg_float("decision_margin_threshold", 20.0),
        axis_length=_cfg_float("axis_length", 0.03),
        t_tag_tcp_target=target_pose,
        rtde_velocity=_cfg_float("rtde_velocity", 0.3),
        rtde_acceleration=_cfg_float("rtde_acceleration", 0.3),
        dry_run=bool(_cfg_bool("dry_run", False) or args.dry_run),
    )
    return cfg


def _load_transform(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到手眼结果文件: {path}")
    mat = np.loadtxt(path, dtype=np.float64)
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"手眼结果矩阵形状错误: {mat.shape}, 期望 (4,4)")
    return mat


def _pose_to_transform(pose: Sequence[float] | np.ndarray) -> np.ndarray:
    pose_np = np.asarray(pose, dtype=np.float64)
    t = pose_np[:3]
    rvec = pose_np[3:6].reshape(3, 1)
    rot, _ = cv2.Rodrigues(rvec)
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    tf[:3, 3] = t
    return tf


def _transform_to_pose(tf: np.ndarray) -> np.ndarray:
    rot = tf[:3, :3]
    t = tf[:3, 3]
    rvec, _ = cv2.Rodrigues(rot)
    return np.concatenate([t, rvec.reshape(3)], axis=0)


def _invert_transform(tf: np.ndarray) -> np.ndarray:
    """Invert homogeneous transform."""
    rot = tf[:3, :3]
    trans = tf[:3, 3]
    inv_tf = np.eye(4, dtype=np.float64)
    inv_tf[:3, :3] = rot.T
    inv_tf[:3, 3] = -rot.T @ trans
    return inv_tf


def _format_pose_xyz_rpy(pose: np.ndarray) -> tuple[str, str]:
    """Format 6D TCP pose into two compact overlay lines."""
    return (
        f"x={pose[0]:+.4f} y={pose[1]:+.4f} z={pose[2]:+.4f}",
        f"rx={pose[3]:+.4f} ry={pose[4]:+.4f} rz={pose[5]:+.4f}",
    )


def _pose_delta_metrics(current_pose: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
    """Return translation (m) and rotation-vector (rad) deltas between two TCP poses."""
    pos_err = float(np.linalg.norm(current_pose[:3] - target_pose[:3]))
    rot_err = float(np.linalg.norm(current_pose[3:6] - target_pose[3:6]))
    return pos_err, rot_err


def _draw_tag_overlay(
    image: np.ndarray,
    detection: Any,
    camera_matrix: np.ndarray,
    axis_length: float,
) -> None:
    corners = np.asarray(detection.corners, dtype=np.float64).reshape(-1, 2)
    center = np.asarray(detection.center, dtype=np.float64).reshape(2)

    poly = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [poly], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.circle(image, tuple(center.astype(np.int32)), 4, (0, 255, 0), -1)

    tag_id = int(getattr(detection, "tag_id", -1))
    margin = float(getattr(detection, "decision_margin", 0.0))
    cv2.putText(
        image,
        f"id={tag_id} dm={margin:.1f}",
        tuple((center + np.array([8.0, -8.0])).astype(np.int32)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    pose_r = np.asarray(getattr(detection, "pose_R"), dtype=np.float64).reshape(3, 3)
    pose_t = np.asarray(getattr(detection, "pose_t"), dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(pose_r)

    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )

    img_points, _ = cv2.projectPoints(
        axis_points,
        rvec,
        pose_t,
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
    )
    pts = img_points.reshape(-1, 2).astype(np.int32)
    origin = tuple(pts[0])
    cv2.line(image, origin, tuple(pts[1]), (0, 0, 255), 2)
    cv2.line(image, origin, tuple(pts[2]), (0, 255, 0), 2)
    cv2.line(image, origin, tuple(pts[3]), (255, 0, 0), 2)


def _draw_frame_overlay(
    image: np.ndarray,
    t_camera_frame: np.ndarray,
    camera_matrix: np.ndarray,
    axis_length: float,
    label: str,
    origin_color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 3,
) -> None:
    if float(t_camera_frame[2, 3]) <= 1e-6:
        return

    rvec, _ = cv2.Rodrigues(t_camera_frame[:3, :3])
    tvec = t_camera_frame[:3, 3].reshape(3, 1)

    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )
    img_points, _ = cv2.projectPoints(
        axis_points,
        rvec,
        tvec,
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
    )
    pts = img_points.reshape(-1, 2).astype(np.int32)
    origin = tuple(pts[0])

    cv2.circle(image, origin, 5, origin_color, -1)
    cv2.line(image, origin, tuple(pts[1]), (0, 0, 255), thickness)
    cv2.line(image, origin, tuple(pts[2]), (0, 255, 0), thickness)
    cv2.line(image, origin, tuple(pts[3]), (255, 0, 0), thickness)
    cv2.putText(
        image,
        label,
        (origin[0] + 8, origin[1] + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        origin_color,
        2,
        cv2.LINE_AA,
    )


def _detect_target_tag(
    detector: Any,
    undistorted_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    cfg: RuntimeConfig,
) -> tuple[list[Any], Optional[Any]]:
    gray = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2GRAY)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=float(cfg.tag_size),
    )

    target_detection: Optional[Any] = None
    for det in detections:
        tag_id = int(getattr(det, "tag_id", -1))
        margin = float(getattr(det, "decision_margin", 0.0))
        has_pose = getattr(det, "pose_R", None) is not None and getattr(det, "pose_t", None) is not None
        if tag_id != cfg.target_tag_id or margin < cfg.decision_margin_threshold or not has_pose:
            continue
        target_detection = det
        break

    return detections, target_detection


def _detection_to_transform(detection: Any) -> np.ndarray:
    rot = np.asarray(getattr(detection, "pose_R"), dtype=np.float64).reshape(3, 3)
    trans = np.asarray(getattr(detection, "pose_t"), dtype=np.float64).reshape(3)
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    tf[:3, 3] = trans
    return tf


def main() -> None:
    args = _parse_args()
    cfg = _build_runtime_config(args)

    t_base_camera: Optional[np.ndarray] = None
    t_camera_base: Optional[np.ndarray] = None
    t_tcp_camera: Optional[np.ndarray] = None
    t_camera_tcp: Optional[np.ndarray] = None

    if cfg.mode == "eye_to_hand":
        t_base_camera = _load_transform(cfg.handeye_file)
        t_camera_base = _invert_transform(t_base_camera)
    else:
        t_tcp_camera = _load_transform(cfg.handeye_file)
        t_camera_tcp = _invert_transform(t_tcp_camera)

    t_tag_tcp_target = _pose_to_transform(cfg.t_tag_tcp_target)

    camera = RealSenseCamera()
    camera.connect()
    if camera.intrinsics is None or camera.dist_coeffs is None:
        camera.disconnect()
        raise RuntimeError("相机内参或畸变参数为空，无法进行去畸变与位姿估计")

    raw_intr = np.asarray(camera.intrinsics, dtype=np.float64)
    raw_dist = np.asarray(camera.dist_coeffs, dtype=np.float64).reshape(-1, 1)

    detector = _create_apriltag_detector(cfg.tag_family)

    rtde_c = None
    rtde_r = None
    if not cfg.dry_run:
        rtde_c, rtde_r = _create_rtde_interfaces(cfg.robot_ip)

    print(f"\n模式: {cfg.mode}")
    print("开始循环: Enter=锁定并运动, c=清除锁定, Esc=退出")
    print(f"目标Tag ID: {cfg.target_tag_id}, 家族: {cfg.tag_family}, tag_size: {cfg.tag_size}m")
    if cfg.dry_run:
        print("当前为 dry-run 模式: 不发送运动命令")

    optimal_intr: Optional[np.ndarray] = None
    live_target_pose: Optional[np.ndarray] = None
    locked_target_pose: Optional[np.ndarray] = None
    actual_tcp_pose: Optional[np.ndarray] = None
    t_base_tag_lock: Optional[np.ndarray] = None

    # 到达判定阈值: 平移 2mm, 旋转向量范数 0.02rad
    pos_arrival_threshold_m = 0.002
    rot_arrival_threshold_rad = 0.02

    try:
        while True:
            color_bgr, _ = camera.get_data()

            h, w = color_bgr.shape[:2]
            if optimal_intr is None:
                optimal_intr, _ = cv2.getOptimalNewCameraMatrix(raw_intr, raw_dist, (w, h), 1.0, (w, h))

            undistorted = cv2.undistort(color_bgr, raw_intr, raw_dist, None, optimal_intr)
            detections, target_det = _detect_target_tag(detector, undistorted, optimal_intr, cfg)

            for det in detections:
                _draw_tag_overlay(undistorted, det, optimal_intr, cfg.axis_length)

            if rtde_r is not None:
                try:
                    actual_tcp_pose = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)
                except Exception:
                    actual_tcp_pose = None
            else:
                actual_tcp_pose = None

            t_base_tcp_actual: Optional[np.ndarray] = None
            if actual_tcp_pose is not None and actual_tcp_pose.shape[0] >= 6:
                t_base_tcp_actual = _pose_to_transform(actual_tcp_pose)

            status_text = "NO TARGET"
            if target_det is not None:
                t_camera_tag = _detection_to_transform(target_det)
                t_camera_tcp_target = t_camera_tag @ t_tag_tcp_target

                if cfg.mode == "eye_to_hand":
                    assert t_base_camera is not None
                    t_base_tcp_target = t_base_camera @ t_camera_tag @ t_tag_tcp_target
                    live_target_pose = _transform_to_pose(t_base_tcp_target)
                else:
                    # Eye-on-Hand: 依赖当前TCP姿态估计 base->tag，再推算目标TCP。
                    if t_base_tcp_actual is not None and t_tcp_camera is not None:
                        t_base_tag_live = t_base_tcp_actual @ t_tcp_camera @ t_camera_tag
                        t_base_tcp_target = t_base_tag_live @ t_tag_tcp_target
                        live_target_pose = _transform_to_pose(t_base_tcp_target)
                    else:
                        live_target_pose = None

                _draw_frame_overlay(
                    undistorted,
                    t_camera_tcp_target,
                    optimal_intr,
                    cfg.axis_length,
                    label="TCP target",
                    origin_color=(255, 255, 255),
                    thickness=3,
                )
                status_text = "TARGET DETECTED"
            else:
                live_target_pose = None

            cv2.putText(
                undistorted,
                f"status: {status_text}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if live_target_pose is not None else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                undistorted,
                "keys: Enter=lock+move, c=clear lock, Esc=quit",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if live_target_pose is not None:
                pose_msg = "live pose: " + np.array2string(live_target_pose, precision=4, suppress_small=True)
                cv2.putText(
                    undistorted,
                    pose_msg,
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            if locked_target_pose is not None:
                lock_msg = "locked pose: " + np.array2string(locked_target_pose, precision=4, suppress_small=True)
                cv2.putText(
                    undistorted,
                    lock_msg,
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if actual_tcp_pose is not None and actual_tcp_pose.shape[0] >= 6:
                assert t_base_tcp_actual is not None
                if cfg.mode == "eye_to_hand":
                    assert t_camera_base is not None
                    t_camera_tcp_actual = t_camera_base @ t_base_tcp_actual
                else:
                    assert t_camera_tcp is not None
                    t_camera_tcp_actual = t_camera_tcp

                _draw_frame_overlay(
                    undistorted,
                    t_camera_tcp_actual,
                    optimal_intr,
                    cfg.axis_length * 0.9,
                    label="TCP actual",
                    origin_color=(0, 255, 255),
                    thickness=2,
                )

                line1, line2 = _format_pose_xyz_rpy(actual_tcp_pose)
                cv2.putText(
                    undistorted,
                    f"actual tcp: {line1}",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    undistorted,
                    f"           {line2}",
                    (20, 168),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if locked_target_pose is not None:
                    pos_err, rot_err = _pose_delta_metrics(actual_tcp_pose, locked_target_pose)
                    arrived = pos_err <= pos_arrival_threshold_m and rot_err <= rot_arrival_threshold_rad
                    arrive_color = (0, 255, 0) if arrived else (0, 180, 255)
                    arrive_text = "ARRIVED" if arrived else "MOVING"
                    cv2.putText(
                        undistorted,
                        f"to lock: dpos={pos_err*1000:.2f}mm drot={rot_err:.3f}rad [{arrive_text}]",
                        (20, 191),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        arrive_color,
                        2,
                        cv2.LINE_AA,
                    )

            window_title = "Eye-to-Hand AprilTag Target Move" if cfg.mode == "eye_to_hand" else "Eye-on-Hand AprilTag Target Move"
            cv2.imshow(window_title, undistorted)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            if key in (13, 10):
                if live_target_pose is None:
                    print("未检测到有效目标Tag，忽略执行命令")
                    continue

                locked_target_pose = live_target_pose.copy()
                print("\n已锁定目标TCP位姿:")
                print(np.array2string(locked_target_pose, precision=6, suppress_small=False))

                if cfg.mode == "eye_on_hand" and target_det is not None and t_base_tcp_actual is not None:
                    assert t_tcp_camera is not None
                    t_camera_tag_lock = _detection_to_transform(target_det)
                    t_base_tag_lock = t_base_tcp_actual @ t_tcp_camera @ t_camera_tag_lock

                if cfg.dry_run:
                    print("dry-run: 已锁定并跳过机器人运动")
                    continue

                assert rtde_c is not None
                q_target = rtde_c.getInverseKinematics(locked_target_pose.tolist())
                if q_target is None or len(q_target) < 6:
                    print("IK求解失败，未执行运动")
                    continue

                # Use asynchronous mode to keep image stream responsive during robot motion.
                ok = rtde_c.moveJ(q_target, cfg.rtde_velocity, cfg.rtde_acceleration, True)
                print(f"moveJ(异步) 发送结果: {ok}")
                if rtde_r is not None:
                    actual_pose = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)
                    print("当前实际TCP位姿:", np.array2string(actual_pose, precision=6, suppress_small=False))

            if key in (ord('c'), ord('C')):
                locked_target_pose = None
                t_base_tag_lock = None
                print("已清除锁定位姿")
    finally:
        if rtde_c is not None:
            try:
                rtde_c.disconnect()
            except Exception:
                pass
        if rtde_r is not None:
            try:
                rtde_r.disconnect()
            except Exception:
                pass
        camera.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
