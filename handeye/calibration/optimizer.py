#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手眼标定优化器
带深度缩放因子的非线性优化
"""

import numpy as np
from scipy import optimize
from scipy.optimize import OptimizeResult
from typing import List, Optional, Tuple

from handeye.calibration.solver_axxb import HandEyeSolver


class HandEyeOptimizer:
    """手眼标定非线性优化器"""

    def __init__(self, mode: str = 'eye_on_hand') -> None:
        """
        初始化优化器

        Args:
            mode: 'eye_on_hand' 或 'eye_to_hand'
        """
        self.mode = mode
        self.solver = HandEyeSolver(mode)

    def optimize(
        self,
        robot_poses: List[np.ndarray],
        camera_data: List[np.ndarray],
        intrinsics: Optional[np.ndarray],
        initial_X: Optional[np.ndarray] = None,
        z_scale_init: float = 1.0,
        z_scale_bounds: Tuple[float, float] = (0.95, 1.05)
    ) -> Tuple[np.ndarray, float, OptimizeResult]:
        """
        优化求解手眼变换

        Args:
            robot_poses: 机器人末端位姿列表 (4x4)
            camera_data: 相机数据 - 可以是:
                - 标定板位姿列表 (4x4)
                - 3D点列表 (N, 3)
            intrinsics: 相机内参
            initial_X: 初始手眼矩阵
            z_scale_init: 初始深度缩放因子
            z_scale_bounds: 深度缩放因子范围

        Returns:
            X: 优化后的手眼矩阵
            z_scale: 优化后的深度缩放因子
            result: 优化结果
        """
        n = len(robot_poses)

        # 初始估计
        if initial_X is None:
            if isinstance(camera_data[0], np.ndarray) and camera_data[0].shape == (4, 4):
                initial_X = self.solver.solve_axxb_svd(robot_poses, camera_data)
            else:
                # 使用点到点方法
                camera_points = [cd[:3] for cd in camera_data]
                R, t = self.solver.solve_rigid_transform(
                    np.array(camera_points),
                    np.array([rp[:3, 3] for rp in robot_poses])
                )
                initial_X = np.eye(4)
                initial_X[:3, :3] = R
                initial_X[:3, 3] = t

        # 参数向量: [x, y, z, rx, ry, rz, z_scale]
        x0 = np.zeros(7)
        x0[:3] = initial_X[:3, 3]
        x0[3:6] = self.solver.log_rot(initial_X[:3, :3])
        x0[6] = z_scale_init

        # 定义目标函数
        def objective(params: np.ndarray) -> float:
            X = self.solver.pose_to_mat(params[:6])
            z_scale = params[6]

            errors: List[np.ndarray] = []

            for i in range(n):
                if isinstance(camera_data[i], np.ndarray):
                    if camera_data[i].shape == (4, 4):
                        # 位姿形式
                        T_cam = camera_data[i].copy()
                        T_cam[2, 3] *= z_scale  # 应用深度缩放

                        # Eye-on-Hand: robot_pose @ X @ T_cam
                        # Eye-to-Hand: robot_pose @ T_cam @ X
                        if self.mode == 'eye_on_hand':
                            T_world = robot_poses[i] @ X @ T_cam
                        else:
                            T_world = self.solver.invert_transform(robot_poses[i]) @ X @ T_cam
                            
                        if i == 0:
                            T_world_ref = T_world
                            T_world_ref_inv = self.solver.invert_transform(T_world_ref)
                            errors.append(np.zeros(6, dtype=np.float64))
                        else:
                            # 相对误差 T_world * T_world_ref^-1 应该接近单位阵
                            E = T_world @ T_world_ref_inv
                            error_pose = self.solver.mat_to_pose(E)
                            errors.append(error_pose)

                    else:
                        # 3D点形式
                        p_cam = camera_data[i].copy()
                        p_cam[2] *= z_scale

                        # 转换到世界坐标
                        p_world = X @ np.append(p_cam, 1)

                        # 与机器人末端位置比较
                        p_robot = robot_poses[i] @ np.append([0, 0, 0], 1)

                        error = p_world[:3] - p_robot[:3]
                        errors.append(np.asarray(error, dtype=np.float64))

            if not errors:
                return 0.0
            errors_arr = np.concatenate(errors)
            return float(np.sum(errors_arr ** 2))

        # 优化
        result = optimize.minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )

        # 提取结果
        X_opt = self.solver.pose_to_mat(result.x[:6])
        z_scale_opt = result.x[6]

        return X_opt, z_scale_opt, result

    # TODO: 目前仅使用了optimize函数，下面的优化方法暂未启用
    def optimize_with_reprojection(
        self,
        robot_poses: List[np.ndarray],
        corners_2d: List[np.ndarray],
        depth_images: List[np.ndarray],
        intrinsics: np.ndarray,
        initial_X: Optional[np.ndarray] = None,
        z_scale_init: float = 1.0
    ) -> Tuple[np.ndarray, float, OptimizeResult]:
        """
        使用重投影误差优化

        Args:
            robot_poses: 机器人末端位姿
            corners_2d: 角点2D坐标列表
            depth_images: 深度图像列表
            intrinsics: 相机内参
            initial_X: 初始手眼矩阵
            z_scale_init: 初始深度缩放

        Returns:
            X: 优化后的手眼矩阵
            z_scale: 优化后的深度缩放因子
            result: 优化结果
        """
        n = len(robot_poses)

        if initial_X is None:
            initial_X = np.eye(4)

        x0 = np.zeros(7)
        x0[:3] = initial_X[:3, 3]
        x0[3:6] = self.solver.log_rot(initial_X[:3, :3])
        x0[6] = z_scale_init

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        def objective(params: np.ndarray) -> float:
            X = self.solver.pose_to_mat(params[:6])
            z_scale = params[6]

            errors: List[np.ndarray] = []

            for i in range(n):
                # 获取深度
                depth = depth_images[i]
                corners = corners_2d[i]

                # 计算每个角点的重投影误差
                for j in range(len(corners)):
                    u = corners[j, 0, 0]
                    v = corners[j, 0, 1]

                    # 获取深度值
                    u_int = int(np.round(u))
                    v_int = int(np.round(v))
                    if 0 <= v_int < depth.shape[0] and 0 <= u_int < depth.shape[1]:
                        z = depth[v_int, u_int] * z_scale
                    else:
                        continue

                    # 相机坐标系下的3D点
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    p_cam = np.array([x, y, z, 1])

                    # 转换到世界坐标
                    if self.mode == 'eye_on_hand':
                        p_world = X @ p_cam
                    else:
                        p_world = X @ robot_poses[i] @ p_cam

                    # 重新投影
                    if p_world[2] > 0:
                        u_proj = fx * p_world[0] / p_world[2] + cx
                        v_proj = fy * p_world[1] / p_world[2] + cy
                        errors.append(np.array([u - u_proj, v - v_proj], dtype=np.float64))

            if not errors:
                return 0.0
            errors_arr = np.array(errors, dtype=np.float64).flatten()
            return float(np.sum(errors_arr ** 2))

        result = optimize.minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000}
        )

        X_opt = self.solver.pose_to_mat(result.x[:6])
        z_scale_opt = result.x[6]

        return X_opt, z_scale_opt, result


def calibrate(
    robot_poses: List[np.ndarray],
    camera_data: List[np.ndarray],
    mode: str = 'eye_on_hand',
    use_optimization: bool = True
) -> Tuple[np.ndarray, float]:
    """
    便捷标定函数

    Args:
        robot_poses: 机器人位姿列表
        camera_data: 相机数据
        mode: 标定模式
        use_optimization: 是否使用优化

    Returns:
        X: 手眼变换矩阵
        z_scale: 深度缩放因子
    """
    optimizer = HandEyeOptimizer(mode)

    if use_optimization:
        X, z_scale, result = optimizer.optimize(
            robot_poses, camera_data, None,
            z_scale_init=1.0
        )
    else:
        solver = HandEyeSolver(mode)
        X = solver.solve_axxb_svd(robot_poses, camera_data)
        z_scale = 1.0

    return X, z_scale
