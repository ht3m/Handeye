#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手眼标定系统 - UR5 + D405 + ArUco (Eye-on-Hand)
"""

import sys
import os
import numpy as np
from typing import Sequence, cast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from device_manager import DeviceManager
from data_collector import CalibDataCollector
from calibration_solver import CalibrationSolver
from error_calculator import ErrorCalculator
from result_visualizer import ResultVisualizer
from config import CALIBRATION_CONFIG


MODE = 'eye_on_hand'
BACKEND = 'aruco'


def main() -> None:
    """主函数"""
    print("=" * 50)
    print(f"手眼标定系统 - UR5 + D405 + ArUco ({'Eye-on-Hand' if MODE == 'eye_on_hand' else 'Eye-to-Hand'})")
    print("=" * 50)

    # 1. 连接设备
    device_mgr = DeviceManager()
    success = device_mgr.connect()

    if not success:
        print("\n设备连接失败，请检查配置后重试")
        return

    robot = device_mgr.get_robot()
    camera = device_mgr.get_camera()
    if robot is None or camera is None:
        print("设备对象为空，请检查连接流程")
        device_mgr.disconnect()
        return

    # 2. 数据采集
    print("\n" + "=" * 50)
    print("数据采集 - ArUco 标定板")
    print("=" * 50)

    collector = CalibDataCollector(robot, camera, MODE, backend=BACKEND)
    collector.collect_loop()

    min_required_cfg = CALIBRATION_CONFIG.get('min_calibration_points', 6)
    min_required = max(6, int(cast(int, min_required_cfg)))
    current_count = len(collector.get_saved_data())
    if current_count < min_required:
        print(f"\n当前有效样本数: {current_count}，少于最小要求: {min_required}")
        print("请继续采集...")
        collector.collect_loop()

    # 3. 标定计算
    solver = CalibrationSolver(
        MODE,
        intrinsics=camera.intrinsics,
        dist_coeffs=getattr(camera, 'dist_coeffs', None),
        backend=BACKEND
    )
    try:
        robot_poses, camera_poses, corners_2d_list, images = solver.load_data(collector)
    except ValueError as e:
        print(f"错误: {e}")
        print("请继续采集数据后重试")
        device_mgr.disconnect()
        return

    result = solver.solve(robot_poses, camera_poses, corners_2d_list)

    # 4. 误差计算
    print("\n" + "=" * 50)
    print("误差计算")
    print("=" * 50)

    error_calc = ErrorCalculator(
        MODE,
        intrinsics=camera.intrinsics,
        dist_coeffs=camera.dist_coeffs,
        backend=BACKEND
    )

    position_errors = error_calc.calculate_position_error(
        robot_poses, camera_poses, result['X'], result['z_scale']
    )
    rotation_errors = error_calc.calculate_rotation_error(
        robot_poses, camera_poses, result['X']
    )
    rotation_errors_deg = np.degrees(rotation_errors)

    error_calc.print_error_report(position_errors, "位置误差报告", unit='m')
    error_calc.print_error_report(rotation_errors_deg, "旋转误差报告 (deg)", unit='deg')

    # 5. 可视化
    print("\n" + "=" * 50)
    print("可视化")
    print("=" * 50)

    visualizer = ResultVisualizer(MODE)

    visualizer.visualize(
        robot_poses, result['X'], result['z_scale'],
        camera_poses=camera_poses,
    )

    visualizer.visualize_position_rotation_errors(
        position_errors,
        rotation_errors_deg,
        pos_unit='mm',
        rot_unit='deg',
        pos_scale=1000.0,
        rot_scale=1.0
    )

    # 6. 清理
    device_mgr.disconnect()

    print("\n" + "=" * 50)
    print("标定完成!")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()