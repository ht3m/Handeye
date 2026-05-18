#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data collection module - UR5 + D405 + ArUco / Checkerboard
Supports teach-by-demo with keyboard workflow:
    - Space: detect target (ArUco marker / checkerboard corners)
    - Enter: save last successful detection
    - Backspace: cancel last successful detection
    - Esc: exit collection loop
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, cast
import numpy as np
import cv2
import threading
import time

# Set Qt font directory before importing cv2 to reduce runtime font warnings.
_qt_font_dir = "/usr/share/fonts/truetype/dejavu"
if os.path.isdir(_qt_font_dir):
    os.environ.setdefault("QT_QPA_FONTDIR", _qt_font_dir)
    os.environ.setdefault("OPENCV_QT_FONTDIR", _qt_font_dir)

from handeye.calibration.feature_extractor import ArucoDetector, CheckerboardExtractor
from handeye.calibration.transforms import mat_to_pose
from handeye.config import ARUCO_CONFIG, CHECKERBOARD_CONFIG, CALIBRATION_CONFIG, get_data_path


class CaptureFrameData(TypedDict, total=False):
    """One captured frame and its detection payload."""

    success: bool
    rgb: np.ndarray
    display_rgb: np.ndarray
    depth: np.ndarray
    corners: Optional[np.ndarray]
    corners_refined: Optional[np.ndarray]
    tag_pose: np.ndarray
    tag_id: int
    tag_corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray


class SavedFrameData(TypedDict):
    """One persisted calibration sample loaded from disk."""

    tcp: np.ndarray
    corners: Optional[np.ndarray]
    tag_pose: Optional[np.ndarray]
    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    index: str


class CalibDataCollector:
    """标定数据采集器 (UR5 + D405 + ArUco)"""

    def __init__(self, robot: Any, camera: Any, mode: str, backend: str = 'aruco') -> None:
        """
        初始化数据采集器

        Args:
            robot: UR5 机器人对象
            camera: D405 相机对象
            mode: 'eye_on_hand' 或 'eye_to_hand'
            backend: 'aruco' 或 'checkerboard'
        """
        self.robot = robot
        self.camera = camera
        self.mode = mode
        self.backend = backend

        # Checkerboard 配置
        cb_size_any = CHECKERBOARD_CONFIG['size']
        cb_size_seq = cast(Sequence[int], cb_size_any)
        cb_cols, cb_rows = int(cb_size_seq[0]), int(cb_size_seq[1])
        cb_square = float(cast(float, CHECKERBOARD_CONFIG['square_size']))
        self.extractor = CheckerboardExtractor((cb_cols, cb_rows), cb_square)
        self.cb_size: Tuple[int, int] = (cb_cols, cb_rows)

        # ArUco 配置
        aruco_dict = str(ARUCO_CONFIG['dictionary'])
        aruco_marker_size = float(cast(float, ARUCO_CONFIG['marker_size']))
        aruco_marker_id = int(cast(int, ARUCO_CONFIG['marker_id']))
        self.aruco_axis_length = max(aruco_marker_size * 0.5, 1e-6)
        self.aruco_detector = ArucoDetector(
            dictionary_name=aruco_dict,
            marker_size=aruco_marker_size,
            marker_id=aruco_marker_id
        )

        # 数据保存路径
        self.data_path = get_data_path(mode)
        self._ensure_dirs()

        # 采集计数器
        self.frame_count = 0

        # 相机内参
        self.intrinsics = camera.intrinsics
        self.dist_coeffs = np.asarray(
            getattr(camera, 'dist_coeffs', np.zeros((5, 1))),
            dtype=np.float64
        ).reshape(-1, 1)

        # 实时预览控制
        self._preview_active = False
        self._preview_thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # 最新有效的检测结果, Space 触发检测, Enter 触发保存
        self._pending_detection: Optional[CaptureFrameData] = None
        self._memory_data: List[SavedFrameData] = []
        min_pts = CALIBRATION_CONFIG.get('min_calibration_points', 6)
        self.min_frames_required = max(6, int(cast(int, min_pts)))

    def _ensure_dirs(self) -> None:
        """确保数据目录存在"""
        os.makedirs(self.data_path['poses'], exist_ok=True)
        os.makedirs(self.data_path['images'], exist_ok=True)

    def clear_old_data(self) -> None:
        """清理旧标定数据"""
        for dir_key in ['poses', 'images']:
            dir_path = self.data_path.get(dir_key)
            if dir_path and os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"警告：无法删除旧文件 {file_path}: {e}")
        print("已清理历史采集数据。")

    def get_frame_index(self) -> int:
        """获取下一个帧索引"""
        existing: List[int] = []
        poses_dir = self.data_path['poses']
        if os.path.exists(poses_dir):
            for f in os.listdir(poses_dir):
                if f.startswith('tcp_') and f.endswith('.txt'):
                    idx = int(f.split('_')[1].split('.')[0])
                    existing.append(idx)

        if existing:
            return max(existing) + 1
        return 1

    # ==================== 采集与检测 ====================

    def capture_and_detect(self) -> 'CaptureFrameData':
        """
        采集一帧并检测目标

        Returns:
            CaptureFrameData: 检测结果
        """
        rgb, depth = self.camera.get_data()
        if rgb is None or depth is None:
            return {
                'success': False,
                'rgb': np.zeros((1, 1, 3), dtype=np.uint8),
                'depth': np.zeros((1, 1)),
                'corners': None,
                'corners_refined': None
            }

        if self.backend == 'aruco':
            return self._capture_and_detect_aruco(rgb, depth)

        # Checkerboard 后端
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        success, corners, corners_refined = self.extractor.detect_corners(gray, refine=True)

        if not success:
            return {
                'success': False,
                'rgb': rgb,
                'depth': depth,
                'corners': None,
                'corners_refined': None
            }

        return {
            'success': True,
            'rgb': rgb,
            'depth': depth,
            'corners': corners,
            'corners_refined': corners_refined
        }

    def _capture_and_detect_aruco(
        self,
        rgb: np.ndarray,
        depth: np.ndarray
    ) -> 'CaptureFrameData':
        """使用 OpenCV ArUco 模块检测标记并估计位姿"""
        result = self.aruco_detector.detect_marker(
            rgb,
            self.intrinsics,
            self.dist_coeffs
        )

        if not result['success']:
            return {
                'success': False,
                'rgb': rgb,
                'depth': depth,
                'corners': None,
                'corners_refined': None
            }

        return {
            'success': True,
            'rgb': rgb,
            'display_rgb': rgb,
            'depth': depth,
            'tag_pose': result['tag_pose'],
            'tag_id': result['tag_id'],
            'tag_corners': result['tag_corners'],
            'rvec': result['rvec'],
            'tvec': result['tvec'],
            'corners': None,
            'corners_refined': None
        }

    # ==================== 保存数据 ====================

    def save_frame(self, frame_data: 'CaptureFrameData') -> bool:
        """
        保存一帧标定数据

        Args:
            frame_data: capture_and_detect 返回的数据

        Returns:
            bool: 是否保存成功
        """
        if not frame_data.get('success', False):
            print("  [X] 目标检测失败，无法保存")
            return False

        idx = self.get_frame_index()

        # 1. 保存 TCP 位姿 (4x4 齐次矩阵)
        tcp_pose = self.robot.get_transform_matrix()
        tcp_xyz_rxyz = mat_to_pose(tcp_pose)
        print(f"  [机械臂TCP] xyz=({tcp_xyz_rxyz[0]:.4f}, {tcp_xyz_rxyz[1]:.4f}, {tcp_xyz_rxyz[2]:.4f}) "
              f"rx={tcp_xyz_rxyz[3]:.4f} ry={tcp_xyz_rxyz[4]:.4f} rz={tcp_xyz_rxyz[5]:.4f}")
        tcp_path = os.path.join(self.data_path['poses'], f'tcp_{idx:03d}.txt')
        np.savetxt(tcp_path, tcp_pose, delimiter=' ')
        print(f"  TCP 位姿已保存: {tcp_path}")

        # 2. 保存 ArUco 检测结果
        if self.backend == 'aruco':
            tag_pose = np.asarray(frame_data.get('tag_pose'), dtype=np.float64)
            if tag_pose.shape != (4, 4):
                print("  [X] ArUco 位姿无效，不能保存")
                return False
            tag_xyz_rxyz = mat_to_pose(tag_pose)
            print(f"  [Tag位姿]   xyz=({tag_xyz_rxyz[0]:.4f}, {tag_xyz_rxyz[1]:.4f}, {tag_xyz_rxyz[2]:.4f}) "
                  f"rx={tag_xyz_rxyz[3]:.4f} ry={tag_xyz_rxyz[4]:.4f} rz={tag_xyz_rxyz[5]:.4f}")
            tag_pose_path = os.path.join(self.data_path['poses'], f'tag_pose_{idx:03d}.txt')
            np.savetxt(tag_pose_path, tag_pose, delimiter=' ')
            print(f"  ArUco 位姿已保存: {tag_pose_path}")

            tag_corners = frame_data.get('tag_corners')
            if tag_corners is not None:
                tag_corners_path = os.path.join(
                    self.data_path['poses'], f'tag_corners_{idx:03d}.txt'
                )
                np.savetxt(
                    tag_corners_path,
                    np.asarray(tag_corners).reshape(-1, 2),
                    delimiter=' '
                )
        else:
            # 保存棋盘格角点
            corners = frame_data.get('corners_refined')
            if corners is None:
                print("  [X] 角点检测数据无效，不能保存")
                return False
            corners_path = os.path.join(self.data_path['poses'], f'corners_{idx:03d}.txt')
            corners_reshaped = corners.reshape(-1, 2)
            np.savetxt(corners_path, corners_reshaped, delimiter=' ')
            print(f"  角点坐标已保存: {corners_path}")

        # 3. 保存 RGB 图像
        rgb_path = os.path.join(self.data_path['images'], f'rgb_{idx:03d}.png')
        cv2.imwrite(rgb_path, frame_data['rgb'])
        print(f"  RGB 图像已保存: {rgb_path}")

        # 4. 保存深度图
        depth_path = os.path.join(self.data_path['images'], f'depth_{idx:03d}.npy')
        np.save(depth_path, frame_data['depth'])
        print(f"  深度图已保存: {depth_path}")

        self.frame_count += 1
        print(f"  [OK] 第 {idx} 帧已保存 (共 {self.frame_count} 帧)")

        return True

    # ==================== 交互操作 ====================

    def save_frame_to_memory(self, frame_data: 'CaptureFrameData') -> bool:
        """Store one accepted sample in memory for evaluation runs."""
        if not frame_data.get('success', False):
            print("  [X] Target detection failed; cannot store sample.")
            return False

        tcp_pose = self.robot.get_transform_matrix()
        tag_pose: Optional[np.ndarray] = None
        if self.backend == 'aruco':
            tag_pose_np = np.asarray(frame_data.get('tag_pose'), dtype=np.float64)
            if tag_pose_np.shape != (4, 4):
                print("  [X] ArUco pose is invalid; cannot store sample.")
                return False
            tag_pose = tag_pose_np

        idx = f"{len(self._memory_data) + 1:03d}"
        self._memory_data.append({
            'tcp': tcp_pose,
            'corners': frame_data.get('corners_refined'),
            'tag_pose': tag_pose,
            'rgb': frame_data.get('rgb'),
            'depth': frame_data.get('depth'),
            'index': idx,
        })
        self.frame_count += 1
        print(f"  [OK] Evaluation sample {idx} accepted in memory.")
        return True

    def detect_current_frame(self) -> bool:
        """采集一帧并检测, 显示结果供用户确认"""
        frame_data = self.capture_and_detect()
        if not frame_data['success']:
            self._pending_detection = None
            print("[X] 目标检测失败。请调整位姿/光照后重试 Space。")
            return False

        self._pending_detection = frame_data
        self._show_detection_result(frame_data)
        print("[OK] 目标检测成功。按 Enter 保存当前帧。")
        return True

    def _show_detection_result(self, frame_data: 'CaptureFrameData') -> None:
        """显示检测结果可视化"""
        rgb = frame_data.get('display_rgb', frame_data['rgb']).copy()

        if self.backend == 'aruco':
            tag_corners = frame_data.get('tag_corners')
            tag_id = frame_data.get('tag_id', -1)
            if tag_corners is None:
                return
            pts = np.asarray(tag_corners, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(rgb, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            center = np.mean(pts, axis=0).astype(int)
            cv2.circle(rgb, tuple(center), 6, (0, 0, 255), -1)
            rvec = frame_data.get('rvec')
            tvec = frame_data.get('tvec')
            tag_pose = frame_data.get('tag_pose')
            if (rvec is None or tvec is None) and tag_pose is not None:
                tag_pose_np = np.asarray(tag_pose, dtype=np.float64)
                if tag_pose_np.shape == (4, 4):
                    rvec, _ = cv2.Rodrigues(tag_pose_np[:3, :3])
                    tvec = tag_pose_np[:3, 3].reshape(3, 1)
            if rvec is not None and tvec is not None:
                cv2.drawFrameAxes(
                    rgb,
                    self.intrinsics,
                    self.dist_coeffs,
                    np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                    np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                    self.aruco_axis_length,
                    3
                )
            cv2.putText(
                rgb,
                f'ArUco ID={tag_id}',
                (int(center[0]) + 10, int(center[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            cv2.putText(
                rgb,
                'Enter=save  Backspace=cancel  Space=detect again  Esc=exit',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.imshow('ArUco Detection Result', rgb)
            cv2.waitKey(1)
            return

        # Checkerboard 可视化
        corners = frame_data.get('corners_refined')
        if corners is None:
            return

        cv2.drawChessboardCorners(rgb, self.cb_size, corners, True)

        if corners is not None and len(corners) > 0:
            start_pt = tuple(corners[0, 0].astype(int))
            end_pt = tuple(corners[-1, 0].astype(int))
            cv2.circle(rgb, start_pt, 9, (0, 255, 255), -1)
            cv2.putText(rgb, 'START', (start_pt[0] + 8, start_pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(rgb, end_pt, 9, (255, 0, 255), -1)
            cv2.putText(rgb, 'END', (end_pt[0] + 8, end_pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imshow('Checkerboard Detection Result', rgb)
        cv2.waitKey(100)

    # ==================== 预览与采集循环 ====================

    def _preview_loop(self) -> None:
        """后台线程: 实时相机图像显示"""
        cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)

        while self._preview_active:
            try:
                rgb, _depth = self.camera.get_data()

                if rgb is None or rgb.size == 0 or rgb.mean() < 1.0:
                    time.sleep(0.05)
                    continue

                with self._frame_lock:
                    self._latest_frame = rgb.copy()

                preview = rgb.copy()
                backend_label = "ArUco" if self.backend == 'aruco' else "Checkerboard"
                target_hint = f"Space: detect {backend_label}  Enter: save  Backspace: cancel  Esc: exit"
                cv2.putText(preview, target_hint, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(
                    preview,
                    f"Collected: {self.frame_count} / Min: {self.min_frames_required}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2
                )
                cv2.imshow('Camera Preview', preview)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Preview error: {e}")
                time.sleep(0.1)

        cv2.destroyWindow('Camera Preview')

    def _update_preview_once(self) -> int:
        """渲染一帧预览并返回按键码"""
        try:
            rgb, _ = self.camera.get_data()
            if rgb is None or rgb.size == 0 or rgb.mean() < 1.0:
                return cv2.waitKey(30) & 0xFF

            with self._frame_lock:
                self._latest_frame = rgb.copy()

            preview = rgb.copy()
            backend_label = "ArUco" if self.backend == 'aruco' else "Checkerboard"
            target_hint = f"Space: detect {backend_label}  Enter: save  Backspace: cancel  Esc: exit"
            cv2.putText(preview, target_hint, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(
                preview,
                f"Collected: {self.frame_count} / Min: {self.min_frames_required}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2
            )
            cv2.imshow('Camera Preview', preview)
        except Exception as e:
            print(f"Preview error: {e}")

        return cv2.waitKey(30) & 0xFF

    def start_preview(self) -> None:
        """启动预览窗口"""
        if self._preview_active:
            return

        self._preview_active = True
        cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
        print("实时预览已启动")

    def stop_preview(self) -> None:
        """停止实时预览"""
        self._preview_active = False
        if self._preview_thread:
            self._preview_thread.join(timeout=1.0)
            self._preview_thread = None
        cv2.destroyAllWindows()
        print("实时预览已停止")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新帧图像"""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def collect_loop(self, save_to_disk: bool = True, enforce_minimum: bool = True) -> int:
        """
        循环采集模式: 反复执行单帧采集直到用户选择退出

        Returns:
            int: 采集的总帧数
        """
        print("\n" + "=" * 50)
        print("开始循环采集")
        print("=" * 50)
        backend_label = "ArUco" if self.backend == 'aruco' else "checkerboard corners"
        print("在每个位置:")
        print("  1. 移动 UR5 到新位置 (示教)")
        print(f"  2. 按 Space 检测 {backend_label}")
        print("  3. 按 Enter 保存当前有效检测")
        print("  4. 按 Esc 退出采集")
        print(f"  最少推荐帧数: {self.min_frames_required}")
        print("  预览窗口持续显示相机画面")
        print("=" * 50)

        # 启动预览窗口
        self.start_preview()

        while True:
            key = self._update_preview_once()

            # Esc 退出采集模式
            if key == 27:
                print("\nESC 按下。退出采集模式。")
                break

            # Space 触发检测预览
            if key == ord(' '):
                self.detect_current_frame()

            # Enter 保存最新有效检测
            elif key in (13, 10):
                if self._pending_detection is None:
                    print("[!] 没有有效检测可保存。请先按 Space 检测。")
                    continue

                saved = (
                    self.save_frame(self._pending_detection)
                    if save_to_disk
                    else self.save_frame_to_memory(self._pending_detection)
                )
                if saved:
                    self._pending_detection = None
                    detection_win = 'ArUco Detection Result' if self.backend == 'aruco' else 'Checkerboard Detection Result'
                    cv2.destroyWindow(detection_win)

            elif key in (8, 127):
                if self._pending_detection is not None:
                    self._pending_detection = None
                    detection_win = 'ArUco Detection Result' if self.backend == 'aruco' else 'Checkerboard Detection Result'
                    cv2.destroyWindow(detection_win)
                    print("[OK] Current detection canceled. Preview is active.")
                else:
                    print("[!] No pending detection to cancel.")

            time.sleep(0.01)

        self.stop_preview()

        print(f"\n采集完成，共 {self.frame_count} 帧")
        if enforce_minimum and self.frame_count < self.min_frames_required:
            print(f"[!] 采集帧数 < 推荐最小值 ({self.min_frames_required})。")
        return self.frame_count

    # ==================== 数据读取 ====================

    def get_memory_data(self) -> List['SavedFrameData']:
        """Return samples accepted in memory during an evaluation run."""
        return list(self._memory_data)

    def get_saved_data(self) -> List['SavedFrameData']:
        """
        获取所有已保存的数据

        Returns:
            list: [{'tcp': 4x4 矩阵, 'corners': 角点, 'tag_pose': 标记位姿, ...}, ...]
        """
        data: List[SavedFrameData] = []
        poses_dir = self.data_path['poses']
        images_dir = self.data_path['images']

        if not os.path.exists(poses_dir):
            return data

        tcp_files = sorted([f for f in os.listdir(poses_dir) if f.startswith('tcp_')])

        for tcp_file in tcp_files:
            idx = tcp_file.split('_')[1].split('.')[0]

            # 加载 TCP 位姿
            tcp_path = os.path.join(poses_dir, tcp_file)
            tcp = np.loadtxt(tcp_path)

            # 加载角点
            corners_path = os.path.join(poses_dir, f'corners_{idx}.txt')
            if os.path.exists(corners_path):
                corners = np.loadtxt(corners_path).reshape(-1, 1, 2)
            else:
                corners = None

            # 加载 ArUco 位姿
            tag_pose_path = os.path.join(poses_dir, f'tag_pose_{idx}.txt')
            if os.path.exists(tag_pose_path):
                tag_pose = np.loadtxt(tag_pose_path)
            else:
                tag_pose = None

            # 加载图像
            rgb_path = os.path.join(images_dir, f'rgb_{idx}.png')
            rgb = cv2.imread(rgb_path) if os.path.exists(rgb_path) else None

            depth_path = os.path.join(images_dir, f'depth_{idx}.npy')
            depth = np.load(depth_path) if os.path.exists(depth_path) else None

            data.append({
                'tcp': tcp,
                'corners': corners,
                'tag_pose': tag_pose,
                'rgb': rgb,
                'depth': depth,
                'index': idx
            })

        return data
