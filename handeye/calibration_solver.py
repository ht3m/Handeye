#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标定计算模块 - 负责 AX=XB 求解
"""

import numpy as np
import os
import cv2
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, cast

from handeye.calibration.solver_axxb import HandEyeSolver
from handeye.calibration.optimizer import HandEyeOptimizer
from handeye.calibration.transforms import invert_transform
from handeye.config import CHECKERBOARD_CONFIG, CALIBRATION_CONFIG, get_results_path, REALSENSE_CONFIG


class DataCollectorProtocol(Protocol):
    """Minimal collector interface consumed by CalibrationSolver."""

    intrinsics: np.ndarray
    dist_coeffs: np.ndarray

    def get_saved_data(self) -> Sequence[Mapping[str, Any]]:
        ...


class CalibrationSolver:
    """手眼标定求解器"""

    def __init__(
        self,
        mode: str,
        intrinsics: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        backend: str = 'checkerboard'
    ) -> None:
        """
        初始化求解器

        Args:
            mode: 'eye_on_hand' 或 'eye_to_hand'
        """
        self.mode = mode
        self.backend = backend
        self.solver = HandEyeSolver(mode)
        self.optimizer = HandEyeOptimizer(mode)

        # 相机内参（优先使用外部传入的真实内参）
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
        min_pts = CALIBRATION_CONFIG.get('min_calibration_points', 6)
        self.min_points_required = max(6, int(cast(int, min_pts)))

        if dist_coeffs is not None:
            self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def _build_checkerboard_object_points(self) -> np.ndarray:
        """Build checkerboard object points in board coordinate system."""
        cb_size_any = CHECKERBOARD_CONFIG['size']
        cb_size_tuple = cast(tuple[int, int], cb_size_any)
        cb_cols = int(cb_size_tuple[0])
        cb_rows = int(cb_size_tuple[1])
        square = float(cast(float, CHECKERBOARD_CONFIG['square_size']))
        objp = np.zeros((cb_cols * cb_rows, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2)
        objp *= square
        return objp

    @staticmethod
    def _rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Convert OpenCV rvec/tvec to 4x4 homogeneous transform."""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)
        return T

    @staticmethod
    def _invert_transform(T: np.ndarray) -> np.ndarray:
        """Invert 4x4 homogeneous transform."""
        return invert_transform(T)

    @staticmethod
    def _average_transforms(transforms: List[np.ndarray]) -> np.ndarray:
        """Average multiple 4x4 transforms (SVD for rotation, mean for translation)."""
        if len(transforms) == 0:
            raise ValueError("无法对空变换列表求平均")

        rotations = np.array([T[:3, :3] for T in transforms])
        translations = np.array([T[:3, 3] for T in transforms])

        R_mean = np.mean(rotations, axis=0)
        U, _, Vt = np.linalg.svd(R_mean)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        T_avg = np.eye(4)
        T_avg[:3, :3] = R
        T_avg[:3, 3] = np.mean(translations, axis=0)
        return T_avg

    def _save_pnp_debug_data(
        self,
        frame_index: str,
        objp: np.ndarray,
        corners_2d: np.ndarray,
        camera_pose: np.ndarray
    ) -> None:
        """Save per-frame PnP inputs/outputs for offline debugging."""
        debug_dir = os.path.join(get_results_path(self.mode), 'pnp_debug')
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f'pnp_{frame_index}.npz')
        np.savez(
            debug_path,
            objp=np.asarray(objp, dtype=np.float32),
            corners_2d=np.asarray(corners_2d, dtype=np.float32),
            camera_pose=np.asarray(camera_pose, dtype=np.float64),
        )

    def load_data(
        self,
        data_collector: DataCollectorProtocol
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        加载采集的数据

        Args:
            data_collector: CalibDataCollector 对象

        Returns:
            robot_poses: 机器人位姿列表 (T_base_tcp)
            camera_poses: 相机观测位姿列表 (T_cam_board)
            corners_2d_list: 每帧角点像素坐标
            images: 与有效帧对齐的 RGB 图像列表
        """
        saved_data = data_collector.get_saved_data()

        # 若采集器携带实时相机内参，则覆盖默认内参。
        collector_intrinsics = getattr(data_collector, 'intrinsics', None)
        if collector_intrinsics is not None:
            collector_intrinsics_np = np.asarray(collector_intrinsics, dtype=np.float64)
            if collector_intrinsics_np.shape == (3, 3):
                self.intrinsics = collector_intrinsics_np
        collector_dist = getattr(data_collector, 'dist_coeffs', None)
        if collector_dist is not None:
            collector_dist_np = np.asarray(collector_dist, dtype=np.float64).reshape(-1, 1)
            if collector_dist_np.shape[0] >= 4:
                self.dist_coeffs = collector_dist_np

        if len(saved_data) < self.min_points_required:
            raise ValueError(
                f"数据点不足: {len(saved_data)}, 需要至少{self.min_points_required}个"
            )

        print(f"加载了 {len(saved_data)} 个数据点")

        robot_poses = []
        camera_poses = []
        corners_2d_list = []
        images = []
        objp = self._build_checkerboard_object_points() if self.backend == 'checkerboard' else None
        dist_coeffs = self.dist_coeffs

        for d_any in saved_data:
            d = cast(Dict[str, Any], d_any)
            tcp = d['tcp']
            if self.backend in ('aruco', 'apriltag'):
                tag_pose = d.get('tag_pose')
                if tag_pose is None:
                    print(f"  警告: 帧 {d['index']} 缺少 tag_pose，跳过")
                    continue
                camera_pose = np.asarray(tag_pose, dtype=np.float64)
                if camera_pose.shape != (4, 4):
                    print(f"  警告: 帧 {d['index']} tag_pose 维度异常，跳过")
                    continue
                corners_2d = np.zeros((0, 2), dtype=np.float32)
            else:
                corners = d['corners']

                if corners is None:
                    print(f"  警告: 帧 {d['index']} 数据不完整，跳过")
                    continue

                corners_2d = corners.reshape(-1, 2).astype(np.float32)
                if objp is None:
                    print(f"  警告: 帧 {d['index']} 缺少 checkerboard 点模型，跳过")
                    continue
                ok, rvec, tvec = cv2.solvePnP(
                    objp,
                    corners_2d,
                    self.intrinsics,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if not ok:
                    print(f"  警告: 帧 {d['index']} PnP求解失败，跳过")
                    continue

                camera_pose = self._rvec_tvec_to_transform(rvec, tvec)
                self._save_pnp_debug_data(str(d['index']), objp, corners_2d, camera_pose)

            robot_poses.append(tcp)
            camera_poses.append(camera_pose)
            corners_2d_list.append(corners_2d)
            if d['rgb'] is not None:
                images.append(d['rgb'])
            else:
                images.append(np.zeros((720, 1280, 3), dtype=np.uint8))

        if len(robot_poses) < self.min_points_required:
            raise ValueError(
                f"有效数据不足: {len(robot_poses)}, 需要至少{self.min_points_required}个"
            )

        return robot_poses, camera_poses, corners_2d_list, images

    def solve(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        corners_2d_list: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        执行标定求解

        Args:
            robot_poses: 机器人位姿列表
            camera_poses: 相机观测位姿列表
            corners_2d_list: 角点像素坐标列表

        Returns:
            result: {
                'X': 手眼变换矩阵,
                'z_scale': 深度缩放因子,
                'optimization_result': 优化结果
            }
        """
        print("\n" + "=" * 50)
        print("标定计算")
        print("=" * 50)

        # 1. SVD 求解
        print("\n[1/2] SVD 求解...")
        if self.mode == 'eye_on_hand':
            # Eye-on-Hand:
            camera_poses_for_solver = [self._invert_transform(T) for T in camera_poses]
            X_svd = self.solver.solve_axxb_svd(robot_poses, camera_poses)
        else:
            # Eye-to-Hand:
            X_svd = self.solver.solve_axxb_svd(robot_poses, camera_poses)
        print("SVD 求解完成")

        # 打印SVD结果
        pos = X_svd[:3, 3]
        print(f"  位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

        # 2. 非线性优化
        print("\n[2/2] 非线性优化...")
        try:
            X_opt, z_scale, opt_result = self.optimizer.optimize(
                robot_poses=robot_poses,
                camera_data=camera_poses,
                intrinsics=self.intrinsics,
                initial_X=X_svd,
                z_scale_init=1.0
            )
            print("非线性优化完成")
            print(f"  优化是否成功: {opt_result.success}")
            print(f"  最终目标函数值 (误差): {opt_result.fun:.6e}")
        except Exception as e:
            print(f"优化过程出现异常: {e}")
            X_opt = X_svd
            z_scale = 1.0
            opt_result = None
            print("回退使用SVD结果作为最终解")

        print(f"  深度缩放因子: {z_scale:.6f}")

        # 打印优化后结果
        pos = X_opt[:3, 3]
        print(f"  位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

        # 保存结果
        self._save_result(X_opt, z_scale)

        return {
            'X': X_opt,
            'z_scale': z_scale,
            'optimization_result': opt_result,
            'robot_poses': robot_poses,
            'camera_poses': camera_poses,
            'corners_2d_list': corners_2d_list
        }

    def _save_result(self, X: np.ndarray, z_scale: float) -> None:
        """保存结果到文件"""
        results_path = get_results_path(self.mode)
        os.makedirs(results_path, exist_ok=True)

        # 保存手眼变换矩阵
        np.savetxt(os.path.join(results_path, 'handeye_transform.txt'), X, delimiter=' ')

        # 保存深度缩放因子
        np.savetxt(os.path.join(results_path, 'depth_scale.txt'), np.array([z_scale]), delimiter=' ')

        # 保存详细信息
        with open(os.path.join(results_path, 'calibration_info.txt'), 'w') as f:
            f.write(f"Mode: {self.mode}\n")
            f.write(f"Z-scale: {z_scale:.6f}\n")
            f.write(f"\nTransform (4x4):\n")
            f.write(str(X))

        print(f"\n结果已保存至: {results_path}")
