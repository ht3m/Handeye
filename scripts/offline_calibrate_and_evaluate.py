#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline hand-eye calibration and evaluation from copied data folders."""

import os
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from handeye.calibration_solver import CalibrationSolver  # noqa: E402
from handeye.config import (  # noqa: E402
    CALIBRATION_BACKEND,
    CALIBRATION_MODE,
    get_data_path,
    validate_calibration_settings,
)
from handeye.error_calculator import ErrorCalculator  # noqa: E402


class OfflineDataCollector:
    """Minimal collector interface for data already saved under data/{mode}."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.data_path = get_data_path(mode)
        self.intrinsics: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

    def get_saved_data(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        poses_dir = self.data_path['poses']
        images_dir = self.data_path['images']

        if not os.path.isdir(poses_dir):
            return data

        tcp_files = sorted(
            name for name in os.listdir(poses_dir)
            if name.startswith('tcp_') and name.endswith('.txt')
        )

        for tcp_file in tcp_files:
            idx = tcp_file.split('_')[1].split('.')[0]
            tcp_path = os.path.join(poses_dir, tcp_file)
            tag_pose_path = os.path.join(poses_dir, f'tag_pose_{idx}.txt')
            corners_path = os.path.join(poses_dir, f'tag_corners_{idx}.txt')
            rgb_path = os.path.join(images_dir, f'rgb_{idx}.png')
            depth_path = os.path.join(images_dir, f'depth_{idx}.npy')

            if not os.path.exists(rgb_path):
                print(f"[WARN] Skip {idx}: missing RGB image, likely a removed old sample")
                continue

            if not os.path.exists(tag_pose_path) and not os.path.exists(corners_path):
                print(f"[WARN] Skip {idx}: missing tag pose/corners data")
                continue

            try:
                tcp = np.loadtxt(tcp_path)
            except Exception as exc:
                print(f"[WARN] Skip {idx}: failed to load TCP pose: {exc}")
                continue

            tag_pose = np.loadtxt(tag_pose_path) if os.path.exists(tag_pose_path) else None
            corners = (
                np.loadtxt(corners_path).reshape(-1, 1, 2)
                if os.path.exists(corners_path)
                else None
            )
            rgb = cv2.imread(rgb_path) if os.path.exists(rgb_path) else None
            depth = np.load(depth_path) if os.path.exists(depth_path) else None

            data.append({
                'tcp': tcp,
                'corners': corners,
                'tag_pose': tag_pose,
                'rgb': rgb,
                'depth': depth,
                'index': idx,
            })

        print(f"Loaded {len(data)} complete offline samples.")
        return data


def main() -> None:
    validate_calibration_settings()
    mode = CALIBRATION_MODE
    backend = CALIBRATION_BACKEND

    print("=" * 50)
    print(f"Offline calibration and evaluation - {mode} / {backend}")
    print("=" * 50)
    print(f"Reading samples from: {get_data_path(mode)['root']}")

    collector = OfflineDataCollector(mode)
    solver = CalibrationSolver(mode, backend=backend)

    robot_poses, camera_poses, corners_2d_list, _images = solver.load_data(collector)
    result = solver.solve(robot_poses, camera_poses, corners_2d_list)

    error_calc = ErrorCalculator(mode, backend=backend)
    position_errors = error_calc.calculate_position_error(
        robot_poses,
        camera_poses,
        result['X'],
        result['z_scale'],
    )
    rotation_errors_deg = np.degrees(
        error_calc.calculate_rotation_error(robot_poses, camera_poses, result['X'])
    )

    error_calc.print_error_report(position_errors, "Position consistency error", unit='m')
    error_calc.print_error_report(rotation_errors_deg, "Rotation consistency error", unit='deg')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOffline run interrupted.")
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback

        traceback.print_exc()
