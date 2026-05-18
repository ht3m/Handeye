#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手眼标定系统配置文件 - UR5 + D405 + ArUco
"""

import numpy as np
import os
from typing import Dict, List, Optional

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# UR5 机械臂配置
UR5_CONFIG = {
    'tcp_host_ip': '169.254.162.111',  # UR5 IP地址
    'tcp_port': 30003,
    'workspace_limits': None,
    'default_velocity': 1.05,
    'default_acceleration': 1.4,
}

# RealSense D405 相机配置
REALSENSE_CONFIG = {
    'device_id': None,  # 自动选择第一个设备
    'width': 1280,
    'height': 720,
    'fps': 30,
    # D405 内参
    'default_intrinsics': {
        'fx': 591.3669592841128, 'fy': 590.1281027314918,
        'cx': 643.8559356720739, 'cy': 370.3876182383728,
    }
}

# ArUco 标定板配置
ARUCO_CONFIG = {
    'dictionary': 'DICT_ARUCO_ORIGINAL',  # ArUco 字典类型 (OpenCV 5x5)
    'marker_id': 996,                      # 目标 ArUco 标记 ID
    'marker_size': 0.10,                   # 标记边长 (米)
}

# 标定板配置 (保留棋盘格配置供兼容)
CHECKERBOARD_CONFIG = {
    'size': (11, 8),  # 内角点数量 (cols, rows)
    'square_size': 0.006,  # 棋盘格方格大小 (米)
    'board_to_base_rough': [-0.0553, -0.3491, 0.0437, -70.5, -163.36, 7.45],
    'board_to_tcp_rough': [0.005, 0, 0.075, 0, -90, 0],
}

# 标定模式配置 (仅 Eye-on-Hand)
CALIBRATION_MODES = ['eye_on_hand', 'eye_to_hand']
CALIBRATION_MODE = 'eye_on_hand'  # eye_on_hand: 眼在手上; eye_to_hand: 眼在手外 / hand-to-eye
CALIBRATION_BACKEND = 'aruco'

# SVD 采集与特征分析配置
SVD_CONFIG = {
    'data_root': os.path.join(PROJECT_ROOT, 'data', 'svd'),
    'images_dirname': 'images',
    'features_filename': 'svd_features.csv',
}

# 标定参数配置
CALIBRATION_CONFIG = {
    'z_scale_init': 1.0,  # 深度缩放因子初始值
    'z_scale_bounds': (0.95, 1.05),  # 深度缩放因子搜索范围
    'optimization_method': 'Nelder-Mead',
    'min_calibration_points': 6,  # 最少标定点数
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'coordinate_axis_length': 0.1,  # 坐标系轴长度 (米)
    'show_checkerboard': True,
    'show_robot': True,
    'figsize': (12, 10),
}

# 数据路径配置
def get_data_path(mode: str) -> Dict[str, str]:
    """获取指定模式的数据路径"""
    base_path = os.path.join(PROJECT_ROOT, 'data', mode)
    return {
        'root': base_path,
        'teach_poses': os.path.join(base_path, 'teach_poses'),
        'poses': os.path.join(base_path, 'poses'),
        'images': os.path.join(base_path, 'images'),
    }

def get_results_path(mode: str) -> str:
    """获取指定模式的结果路径"""
    return os.path.join(PROJECT_ROOT, 'results', mode)


def validate_calibration_settings() -> None:
    """Validate user-facing calibration settings before devices are opened."""
    if CALIBRATION_MODE not in CALIBRATION_MODES:
        valid = ', '.join(CALIBRATION_MODES)
        raise ValueError(
            f"Invalid CALIBRATION_MODE={CALIBRATION_MODE!r}. "
            f"Expected one of: {valid}"
        )

    if CALIBRATION_BACKEND != 'aruco':
        raise ValueError(
            f"Invalid CALIBRATION_BACKEND={CALIBRATION_BACKEND!r}. "
            "This main workflow currently supports only 'aruco'."
        )


def get_svd_data_path() -> Dict[str, str]:
    """获取 SVD 采集与分析数据路径。"""
    data_root = str(SVD_CONFIG['data_root'])
    images_dir = os.path.join(data_root, str(SVD_CONFIG['images_dirname']))
    features_path = os.path.join(data_root, str(SVD_CONFIG['features_filename']))
    return {
        'root': data_root,
        'images': images_dir,
        'features': features_path,
    }
