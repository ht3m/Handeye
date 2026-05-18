#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect independent ArUco validation samples and evaluate a saved hand-eye result."""

import os
import sys
from typing import Any, Dict, List, cast

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    CALIBRATION_BACKEND,
    CALIBRATION_MODE,
    get_results_path,
    validate_calibration_settings,
)
from data_collector import CalibDataCollector  # noqa: E402
from device_manager import DeviceManager  # noqa: E402
from error_calculator import ErrorCalculator  # noqa: E402


def _load_transform(mode: str) -> np.ndarray:
    path = os.path.join(get_results_path(mode), 'handeye_transform.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing calibration result: {path}")

    X = np.asarray(np.loadtxt(path), dtype=np.float64)
    if X.shape != (4, 4):
        raise ValueError(f"Invalid hand-eye transform shape: {X.shape}; expected (4, 4)")
    return X


def _load_z_scale(mode: str) -> float:
    path = os.path.join(get_results_path(mode), 'depth_scale.txt')
    if not os.path.exists(path):
        return 1.0
    values = np.asarray(np.loadtxt(path), dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 1.0
    return float(values[0])


def _split_samples(samples: List[Dict[str, Any]]) -> tuple[List[np.ndarray], List[np.ndarray]]:
    robot_poses: List[np.ndarray] = []
    camera_poses: List[np.ndarray] = []

    for sample in samples:
        tcp = np.asarray(sample.get('tcp'), dtype=np.float64)
        tag_pose = sample.get('tag_pose')
        if tcp.shape != (4, 4) or tag_pose is None:
            continue

        cam_pose = np.asarray(tag_pose, dtype=np.float64)
        if cam_pose.shape != (4, 4):
            continue

        robot_poses.append(tcp)
        camera_poses.append(cam_pose)

    return robot_poses, camera_poses


def main() -> None:
    validate_calibration_settings()
    mode = CALIBRATION_MODE
    backend = CALIBRATION_BACKEND

    X = _load_transform(mode)
    z_scale = _load_z_scale(mode)
    print("=" * 50)
    print(f"Calibration evaluation - {mode} / {backend}")
    print("=" * 50)
    print(f"Loaded transform from: {os.path.join(get_results_path(mode), 'handeye_transform.txt')}")

    device_mgr = DeviceManager()
    if not device_mgr.connect():
        print("Device connection failed.")
        return

    try:
        robot = device_mgr.get_robot()
        camera = device_mgr.get_camera()
        if robot is None or camera is None:
            print("Robot or camera is unavailable.")
            return

        collector = CalibDataCollector(robot, camera, mode, backend=backend)
        collector.min_frames_required = 0
        print("\nCollect validation samples. Space=detect, Enter=accept, Backspace=cancel, Esc=finish.")
        collector.collect_loop(save_to_disk=False, enforce_minimum=False)

        samples = cast(List[Dict[str, Any]], collector.get_memory_data())
        robot_poses, camera_poses = _split_samples(samples)
        if not robot_poses:
            print("No valid evaluation samples collected.")
            return

        error_calc = ErrorCalculator(
            mode,
            intrinsics=camera.intrinsics,
            dist_coeffs=camera.dist_coeffs,
            backend=backend,
        )

        position_errors = error_calc.calculate_position_error(
            robot_poses,
            camera_poses,
            X,
            z_scale,
        )
        rotation_errors_deg = np.degrees(
            error_calc.calculate_rotation_error(robot_poses, camera_poses, X)
        )

        error_calc.print_error_report(position_errors, "Position consistency error", unit='m')
        error_calc.print_error_report(rotation_errors_deg, "Rotation consistency error", unit='deg')
    finally:
        device_mgr.disconnect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback

        traceback.print_exc()
