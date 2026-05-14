#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
误差计算模块 - 计算重投影误差
"""

import numpy as np
import cv2
import sys
import os
from typing import Dict, List, Optional, cast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CHECKERBOARD_CONFIG, REALSENSE_CONFIG
from calibration.transforms import invert_transform, pose_to_mat


class ErrorCalculator:
    """重投影误差计算器"""

    def __init__(
        self,
        mode: str,
        intrinsics: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        backend: str = 'checkerboard'
    ) -> None:
        """
        初始化误差计算器

        Args:
            mode: 'eye_on_hand' 或 'eye_to_hand'
            intrinsics: 相机内参 (3x3)
            dist_coeffs: 相机畸变参数
        """
        self.mode = mode
        self.backend = backend

        # 相机内参（优先使用外部传入真实内参）
        if intrinsics is not None:
            self.intrinsics = np.asarray(intrinsics, dtype=np.float64)
        else:
            intr_any = REALSENSE_CONFIG.get('default_intrinsics')
            intr = cast(Dict[str, float], intr_any)
            self.intrinsics = np.array([
                [float(intr['fx']), 0, float(intr['cx'])],
                [0, float(intr['fy']), float(intr['cy'])],
                [0, 0, 1]
            ], dtype=np.float64)
            
        # 相机畸变参数（如果未提供，则默认为0）
        if dist_coeffs is not None:
            self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # 棋盘格参数
        cb_size_any = CHECKERBOARD_CONFIG['size']
        cb_size_tuple = cast(tuple[int, int], cb_size_any)
        self.cb_size: tuple[int, int] = (int(cb_size_tuple[0]), int(cb_size_tuple[1]))
        self.square_size: float = float(cast(float, CHECKERBOARD_CONFIG['square_size']))

    def _build_checkerboard_object_points(self) -> np.ndarray:
        """Build checkerboard corner points in board coordinate frame."""
        cb_cols, cb_rows = self.cb_size
        objp = np.zeros((cb_cols * cb_rows, 3), dtype=np.float64)
        objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    @staticmethod
    def _rotation_angle_error_rad(R1: np.ndarray, R2: np.ndarray) -> float:
        """Compute relative rotation angle between two rotation matrices in radians."""
        R_rel = R1 @ R2.T
        cos_theta = (np.trace(R_rel) - 1.0) / 2.0
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        return float(np.arccos(cos_theta))

    def calculate_reprojection_error(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        corners_2d_list: List[np.ndarray],
        X: np.ndarray,
        z_scale: float = 1.0,
        board_to_base: Optional[List[float]] = None,
        board_to_tcp: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        计算重投影误差

        对于每帧数据:
        1. 将标定板角点从相机坐标系转换到世界(基座)坐标系
        2. 将世界坐标投影到像素坐标系
        3. 与检测到的角点比较

        Args:
            robot_poses: 机器人位姿列表
            camera_poses: 相机观测位姿列表 (T_cam_board)
            corners_2d_list: 每帧检测角点 (N, 2)
            X: 手眼变换矩阵
            z_scale: 深度缩放因子
            board_to_base: 标定板相对于基座的位姿 (Eye-on-Hand用)
            board_to_tcp: 标定板相对于TCP的位姿 (Eye-to-Hand用)

        Returns:
            errors: 每帧的重投影误差 (像素)
        """
        if self.backend in ('aruco', 'apriltag'):
            print(f"{self.backend.capitalize()} 后端跳过重投影误差计算")
            return np.array([], dtype=np.float64)

        if not (len(robot_poses) == len(camera_poses) == len(corners_2d_list)):
            raise ValueError("robot_poses/camera_poses/corners_2d_list 长度不一致")

        objp = self._build_checkerboard_object_points()

        frame_errors: List[float] = []

        if self.mode == 'eye_on_hand':
            if board_to_base is not None:
                T_base_board_ref = pose_to_mat(board_to_base)
            else:
                base_boards = [tcp @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
                T_base_board_ref = np.mean(np.stack(base_boards), axis=0)

            for tcp, corners in zip(robot_poses, corners_2d_list):
                T_cam_board_pred = invert_transform(X) @ invert_transform(tcp) @ T_base_board_ref
                R = T_cam_board_pred[:3, :3]
                t = T_cam_board_pred[:3, 3]
                
                pcam = (R @ objp.T + t.reshape(3, 1)).T
                valid = pcam[:, 2] > 1e-8
                if not np.any(valid):
                    continue
                    
                rvec, _ = cv2.Rodrigues(R)
                img_pts, _ = cv2.projectPoints(objp, rvec, t, self.intrinsics, self.dist_coeffs)
                img_pts = img_pts.reshape(-1, 2)
                
                u_proj = img_pts[valid, 0]
                v_proj = img_pts[valid, 1]
                
                det = corners[valid]
                err = np.sqrt((det[:, 0] - u_proj) ** 2 + (det[:, 1] - v_proj) ** 2)
                frame_errors.append(float(np.mean(err)))
        else:
            if board_to_tcp is not None:
                T_tcp_board_ref = pose_to_mat(board_to_tcp)
            else:
                base_to_tcp_board_list = [invert_transform(tcp) @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
                T_tcp_board_ref = np.mean(np.stack(base_to_tcp_board_list), axis=0)

            for tcp, corners in zip(robot_poses, corners_2d_list):
                T_cam_board_pred = invert_transform(X) @ tcp @ T_tcp_board_ref
                R = T_cam_board_pred[:3, :3]
                t = T_cam_board_pred[:3, 3]
                
                pcam = (R @ objp.T + t.reshape(3, 1)).T
                valid = pcam[:, 2] > 1e-8
                if not np.any(valid):
                    continue
                    
                rvec, _ = cv2.Rodrigues(R)
                img_pts, _ = cv2.projectPoints(objp, rvec, t, self.intrinsics, self.dist_coeffs)
                img_pts = img_pts.reshape(-1, 2)
                
                u_proj = img_pts[valid, 0]
                v_proj = img_pts[valid, 1]
                
                det = corners[valid]
                err = np.sqrt((det[:, 0] - u_proj) ** 2 + (det[:, 1] - v_proj) ** 2)
                frame_errors.append(float(np.mean(err)))

        return np.array(frame_errors, dtype=np.float64)

    def calculate_position_error(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        X: np.ndarray,
        z_scale: float = 1.0,
        board_to_base: Optional[List[float]] = None,
        board_to_tcp: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        计算位置误差 (3D误差)

        将标定板3D点转换到基座坐标系，与:
        - Eye-on-Hand: 粗略估计的标定板位置比较
        - Eye-to-Hand: TCP @ board_to_tcp 比较

        Args:
            robot_poses: 机器人位姿列表
            camera_poses: 相机观测位姿列表 (T_cam_board)
            X: 手眼变换矩阵
            z_scale: 深度缩放因子
            board_to_base: 标定板相对于基座的位姿 (Eye-on-Hand用)
            board_to_tcp: 标定板相对于TCP的位姿 (Eye-to-Hand用)

        Returns:
            errors: 位置误差列表 (米)
        """
        errors: List[float] = []

        for tcp, cam_pose in zip(robot_poses, camera_poses):
            if self.mode == 'eye_on_hand':
                T_measured = tcp @ X @ cam_pose
                p_measured = T_measured[:3, 3]

                # 与粗略估计比较
                if board_to_base is not None:
                    T_expected = pose_to_mat(board_to_base)
                    p_expected = T_expected[:3, 3]
                    error = float(np.linalg.norm(p_measured - p_expected))
                    errors.append(error)
                else:
                    errors.append(float(0.0))
            else:
                T_measured = X @ cam_pose
                p_measured = T_measured[:3, 3]

                # 与粗略估计比较
                if board_to_tcp is not None:
                    T_board_tcp = pose_to_mat(board_to_tcp)
                    T_expected = tcp @ T_board_tcp
                    p_expected = T_expected[:3, 3]
                    error = float(np.linalg.norm(p_measured - p_expected))
                    errors.append(error)
                else:
                    errors.append(float(0.0))

        return np.array(errors)

    def calculate_rotation_error(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        X: np.ndarray,
        board_to_base: Optional[List[float]] = None,
        board_to_tcp: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        计算旋转误差（相对角，单位 rad）

        对比对象与位置误差保持一致：
        - Eye-on-Hand: 将观测到的标定板姿态(T_base_board)与 board_to_base 粗略姿态比较。
                - Eye-to-Hand: 将观测到的标定板姿态(T_base_board)与
                    由 board_to_tcp 转换得到的基座系参考姿态(T_base_tcp @ T_board_tcp)比较。

        若未提供粗略姿态，则以观测平均姿态作为参考。
        """
        errors_rad: List[float] = []

        if self.mode == 'eye_on_hand':
            measured_list = [tcp @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
            if board_to_base is not None:
                T_ref = pose_to_mat(board_to_base)
            else:
                T_ref = np.mean(np.stack(measured_list), axis=0)

            R_ref = T_ref[:3, :3]
            for T_meas in measured_list:
                angle_rad = self._rotation_angle_error_rad(T_meas[:3, :3], R_ref)
                errors_rad.append(float(angle_rad))
        else:
            measured_list = [X @ cam_pose for cam_pose in camera_poses]
            if board_to_tcp is not None:
                T_board_tcp = pose_to_mat(board_to_tcp)
                for tcp, T_meas in zip(robot_poses, measured_list):
                    T_ref = tcp @ T_board_tcp
                    angle_rad = self._rotation_angle_error_rad(T_meas[:3, :3], T_ref[:3, :3])
                    errors_rad.append(float(angle_rad))
            else:
                T_ref = np.mean(np.stack(measured_list), axis=0)
                R_ref = T_ref[:3, :3]
                for T_meas in measured_list:
                    angle_rad = self._rotation_angle_error_rad(T_meas[:3, :3], R_ref)
                    errors_rad.append(float(angle_rad))

        return np.array(errors_rad, dtype=np.float64)

    def visualize_reprojection_frames(
        self,
        images: List[np.ndarray],
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        corners_2d_list: List[np.ndarray],
        X: np.ndarray,
        board_to_base: Optional[List[float]] = None,
        board_to_tcp: Optional[List[float]] = None,
        window_name: str = 'Reprojection Frame Viewer'
    ) -> None:
        """
        逐帧显示重投影结果。

        - 绿色圆点: 检测到的角点
        - 红色圆点: 根据标定结果重投影得到的角点
        - Eye-on-Hand: 若给定 board_to_base，使用其作为固定参考；否则使用观测均值参考
        - Eye-to-Hand: 若给定 board_to_tcp，使用其作为 TCP 系固定参考；否则使用观测均值参考
        - 左/右方向键（兼容 Shift+方向键）: 切换帧
        - Esc: 退出
        """
        if self.backend in ('aruco', 'apriltag'):
            print(f'{self.backend.capitalize()} 后端跳过逐帧重投影显示')
            return

        n = min(len(images), len(robot_poses), len(camera_poses), len(corners_2d_list))
        if n == 0:
            print('无可视化数据，跳过逐帧重投影显示')
            return

        objp = self._build_checkerboard_object_points()

        if self.mode == 'eye_on_hand':
            if board_to_base is not None:
                T_base_board_ref = pose_to_mat(board_to_base)
            else:
                base_boards = [tcp @ X @ cam_pose for tcp, cam_pose in zip(robot_poses[:n], camera_poses[:n])]
                T_base_board_ref = np.mean(np.stack(base_boards), axis=0)
        else:
            if board_to_tcp is not None:
                T_tcp_board_ref = pose_to_mat(board_to_tcp)
            else:
                tcp_board_list = [invert_transform(tcp) @ X @ cam_pose for tcp, cam_pose in zip(robot_poses[:n], camera_poses[:n])]
                T_tcp_board_ref = np.mean(np.stack(tcp_board_list), axis=0)

        idx = 0
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            img_src = images[idx]
            if img_src is None:
                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                if img_src.ndim == 2:
                    canvas = cv2.cvtColor(img_src.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                else:
                    canvas = img_src.copy().astype(np.uint8)

            if self.mode == 'eye_on_hand':
                T_cam_board_pred = invert_transform(X) @ invert_transform(robot_poses[idx]) @ T_base_board_ref
            else:
                T_cam_board_pred = invert_transform(X) @ robot_poses[idx] @ T_tcp_board_ref

            R = T_cam_board_pred[:3, :3]
            t = T_cam_board_pred[:3, 3]
            
            pcam = (R @ objp.T + t.reshape(3, 1)).T
            valid = pcam[:, 2] > 1e-8

            if np.any(valid):
                rvec, _ = cv2.Rodrigues(R)
                img_pts, _ = cv2.projectPoints(objp, rvec, t, self.intrinsics, self.dist_coeffs)
                reproj_pts = img_pts.reshape(-1, 2)
                reproj_pts = reproj_pts[valid]
            else:
                reproj_pts = np.zeros((0, 2), dtype=np.float64)

            det = corners_2d_list[idx]
            det_pts = det.reshape(-1, 2)

            for pt in det_pts:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

            if len(det_pts) > 0:
                det_start = tuple(det_pts[0].astype(int))
                det_end = tuple(det_pts[-1].astype(int))
                cv2.circle(canvas, det_start, 8, (0, 255, 255), -1)
                cv2.putText(canvas, 'DET_START', (det_start[0] + 8, det_start[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.circle(canvas, det_end, 8, (255, 0, 255), -1)
                cv2.putText(canvas, 'DET_END', (det_end[0] + 8, det_end[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

            for pt in reproj_pts:
                x_i, y_i = int(pt[0]), int(pt[1])
                if 0 <= x_i < canvas.shape[1] and 0 <= y_i < canvas.shape[0]:
                    cv2.circle(canvas, (x_i, y_i), 4, (0, 0, 255), -1)

            if len(reproj_pts) > 0:
                rep_start = tuple(reproj_pts[0].astype(int))
                rep_end = tuple(reproj_pts[-1].astype(int))
                if 0 <= rep_start[0] < canvas.shape[1] and 0 <= rep_start[1] < canvas.shape[0]:
                    cv2.circle(canvas, rep_start, 8, (0, 165, 255), -1)
                    cv2.putText(canvas, 'REPROJ_START', (rep_start[0] + 8, rep_start[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                if 0 <= rep_end[0] < canvas.shape[1] and 0 <= rep_end[1] < canvas.shape[0]:
                    cv2.circle(canvas, rep_end, 8, (255, 255, 0), -1)
                    cv2.putText(canvas, 'REPROJ_END', (rep_end[0] + 8, rep_end[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            if len(reproj_pts) > 0:
                pair_n = min(len(det_pts), len(reproj_pts))
                frame_err = np.linalg.norm(det_pts[:pair_n] - reproj_pts[:pair_n], axis=1)
                frame_err_text = f'mean err: {float(np.mean(frame_err)):.3f}px'
            else:
                frame_err_text = 'mean err: N/A'

            cv2.putText(
                canvas,
                f'Frame {idx + 1}/{n} | Green: detected | Red: reprojected | {frame_err_text}',
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            cv2.putText(
                canvas,
                'Left/Right: switch frame (Shift compatible) | Esc: exit',
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.imshow(window_name, canvas)
            key_raw = cv2.waitKeyEx(0)
            key = key_raw & 0xFF

            if key == 27:
                break

            # Linux/X11 often reports 81/83 for left/right after mask; Windows uses large codes.
            is_left = key in (81,) or key_raw in (2424832,)
            is_right = key in (83,) or key_raw in (2555904,)

            if is_left:
                idx = (idx - 1) % n
            elif is_right:
                idx = (idx + 1) % n

        cv2.destroyWindow(window_name)

    def compute_statistics(self, errors: np.ndarray) -> Dict[str, float]:
        """
        计算误差统计信息

        Args:
            errors: 误差数组

        Returns:
            dict: 统计信息
        """
        if len(errors) == 0:
            return {'mean': 0, 'max': 0, 'min': 0, 'std': 0}

        return {
            'mean': np.mean(errors),
            'max': np.max(errors),
            'min': np.min(errors),
            'std': np.std(errors),
            'median': np.median(errors)
        }

    def print_error_report(
        self,
        errors: np.ndarray,
        title: str = "重投影误差报告",
        unit: str = "auto"
    ) -> None:
        """
        打印误差报告

        Args:
            errors: 误差数组
            title: 报告标题
            unit: 输出单位, 可选 'auto'/'px'/'m'/'deg'
        """
        stats = self.compute_statistics(errors)

        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)
        print(f"数据点数: {len(errors)}")
        if unit == 'deg':
            print(f"平均误差: {stats['mean']:.3f} deg")
            print(f"最大误差: {stats['max']:.3f} deg")
            print(f"最小误差: {stats['min']:.3f} deg")
            print(f"标准差:   {stats['std']:.3f} deg")
            print(f"中位数:   {stats['median']:.3f} deg")
            mean_metric = stats['mean']
        elif unit == 'px':
            print(f"平均误差: {stats['mean']:.3f} px")
            print(f"最大误差: {stats['max']:.3f} px")
            print(f"最小误差: {stats['min']:.3f} px")
            print(f"标准差:   {stats['std']:.3f} px")
            print(f"中位数:   {stats['median']:.3f} px")
            mean_metric = stats['mean']
        elif unit == 'm':
            print(f"平均误差: {stats['mean']*1000:.3f} mm")
            print(f"最大误差: {stats['max']*1000:.3f} mm")
            print(f"最小误差: {stats['min']*1000:.3f} mm")
            print(f"标准差:   {stats['std']*1000:.3f} mm")
            print(f"中位数:   {stats['median']*1000:.3f} mm")
            mean_metric = stats['mean'] * 1000
        elif stats['mean'] > 0.1:
            print(f"平均误差: {stats['mean']:.3f} px")
            print(f"最大误差: {stats['max']:.3f} px")
            print(f"最小误差: {stats['min']:.3f} px")
            print(f"标准差:   {stats['std']:.3f} px")
            print(f"中位数:   {stats['median']:.3f} px")
            mean_metric = stats['mean']
        else:
            print(f"平均误差: {stats['mean']*1000:.3f} mm")
            print(f"最大误差: {stats['max']*1000:.3f} mm")
            print(f"最小误差: {stats['min']*1000:.3f} mm")
            print(f"标准差:   {stats['std']*1000:.3f} mm")
            print(f"中位数:   {stats['median']*1000:.3f} mm")
            mean_metric = stats['mean'] * 1000

        # 评级
        if mean_metric < 2:
            rating = "Excellent (优秀)"
        elif mean_metric < 5:
            rating = "Good (良好)"
        elif mean_metric < 10:
            rating = "Fair (一般)"
        else:
            rating = "Poor (较差)"

        print(f"评级: {rating}")
        print("=" * 50)
