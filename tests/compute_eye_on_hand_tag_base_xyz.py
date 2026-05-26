#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute tag center XYZ in robot base frame for an Eye-on-Hand setup.

Transform convention used by this project:
    T_base_tag = T_base_tcp @ T_tcp_camera @ p_camera_tag

The saved Eye-on-Hand hand-eye result is interpreted as T_tcp_camera.
By default the current robot TCP pose is read automatically from UR5_CONFIG.
Robot TCP pose uses the UR format [x, y, z, rx, ry, rz], where translation is
in meters and rotation is a Rodrigues rotation vector in radians.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from typing import Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from handeye.calibration.transforms import pose_to_mat  # noqa: E402
from handeye.config import UR5_CONFIG, get_results_path  # noqa: E402
from handeye.robot.ur_robot import URRobot  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the tag center XYZ in the robot base frame from the current "
            "robot TCP pose, Eye-on-Hand T_tcp_camera, and tag center in camera frame. "
            "If no TCP source is provided, the TCP pose is read from the robot."
        )
    )
    tcp_group = parser.add_mutually_exclusive_group()
    tcp_group.add_argument(
        "--tcp-pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Current TCP pose [x y z rx ry rz]. Position unit is set by --tcp-position-unit.",
    )
    tcp_group.add_argument(
        "--tcp-pose-file",
        help="Text file containing either a 6D TCP pose or a 4x4 T_base_tcp matrix.",
    )
    tcp_group.add_argument(
        "--tcp-from-robot",
        action="store_true",
        help="Read the current TCP pose from the UR robot. This is also the default TCP source.",
    )

    tag_group = parser.add_mutually_exclusive_group()
    tag_group.add_argument(
        "--tag-camera",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Tag center XYZ in the camera frame. Unit is set by --tag-position-unit.",
    )
    tag_group.add_argument(
        "--tag-camera-file",
        help="Text file containing tag center XYZ, or a 4x4 T_camera_tag matrix.",
    )

    parser.add_argument(
        "--handeye-file",
        default=os.path.join(get_results_path("eye_on_hand"), "handeye_transform.txt"),
        help="4x4 Eye-on-Hand transform file. Default: results/eye_on_hand/handeye_transform.txt",
    )
    parser.add_argument(
        "--tcp-position-unit",
        choices=("m", "mm"),
        default="m",
        help="Translation unit for --tcp-pose or 6D --tcp-pose-file. Default: m.",
    )
    parser.add_argument(
        "--tag-position-unit",
        choices=("m", "mm"),
        default="m",
        help="Translation unit for --tag-camera or vector --tag-camera-file. Default: m.",
    )
    parser.add_argument(
        "--robot-ip",
        default=str(UR5_CONFIG.get("tcp_host_ip", "169.254.162.111")),
        help="UR robot IP used when reading the current TCP pose.",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=int(UR5_CONFIG.get("tcp_port", 30003)),
        help="UR robot TCP port used when reading the current TCP pose.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only base XYZ in meters as three space-separated numbers.",
    )
    return parser.parse_args()


def _position_scale(unit: str) -> float:
    if unit == "m":
        return 1.0
    if unit == "mm":
        return 0.001
    raise ValueError(f"Unsupported unit: {unit}")


def _load_array(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        arr = np.loadtxt(path, dtype=np.float64)
    except ValueError:
        with open(path, "r", encoding="utf-8") as f:
            arr = np.asarray(ast.literal_eval(f.read()), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def _load_transform(path: str, name: str) -> np.ndarray:
    mat = _load_array(path)
    if mat.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got shape {mat.shape}: {path}")
    if not np.allclose(mat[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError(f"{name} bottom row must be [0, 0, 0, 1]: {path}")
    return mat


def _pose_vector_to_transform(pose: Sequence[float], position_unit: str) -> np.ndarray:
    pose_arr = np.asarray(pose, dtype=np.float64).reshape(6).copy()
    pose_arr[:3] *= _position_scale(position_unit)
    return pose_to_mat(pose_arr)


def _read_numbers_from_prompt(prompt: str, expected_count: int) -> np.ndarray:
    while True:
        raw = input(prompt).strip().replace(",", " ")
        try:
            values = np.fromstring(raw, sep=" ", dtype=np.float64)
        except ValueError:
            values = np.array([], dtype=np.float64)
        if values.shape == (expected_count,):
            return values
        print(f"Expected {expected_count} numbers, got {values.size}.")


def _resolve_base_tcp(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.tcp_pose is not None:
        return _pose_vector_to_transform(args.tcp_pose, args.tcp_position_unit), "command line --tcp-pose"

    if args.tcp_pose_file:
        arr = _load_array(args.tcp_pose_file)
        if arr.shape == (4, 4):
            return arr, args.tcp_pose_file
        if arr.size == 6:
            return _pose_vector_to_transform(arr.reshape(6), args.tcp_position_unit), args.tcp_pose_file
        raise ValueError(
            f"TCP pose file must contain 6 values or a 4x4 matrix, got shape {arr.shape}: {args.tcp_pose_file}"
        )

    robot = URRobot(tcp_host_ip=args.robot_ip, tcp_port=args.robot_port)
    tcp_pose = robot.get_tool_pose()
    return pose_to_mat(tcp_pose), f"robot {args.robot_ip}:{args.robot_port}"


def _resolve_camera_tag_point(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    scale = _position_scale(args.tag_position_unit)

    if args.tag_camera is not None:
        return np.asarray(args.tag_camera, dtype=np.float64).reshape(3) * scale, "command line --tag-camera"

    if args.tag_camera_file:
        arr = _load_array(args.tag_camera_file)
        if arr.shape == (4, 4):
            return np.asarray(arr[:3, 3], dtype=np.float64), args.tag_camera_file
        if arr.size == 3:
            return np.asarray(arr, dtype=np.float64).reshape(3) * scale, args.tag_camera_file
        raise ValueError(
            f"Tag camera file must contain 3 values or a 4x4 matrix, got shape {arr.shape}: {args.tag_camera_file}"
        )

    values = _read_numbers_from_prompt(
        "Enter tag center in camera frame x y z (meters unless --tag-position-unit mm): ",
        3,
    )
    return values * scale, "interactive input"


def _transform_point(transform: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    transformed = transform @ point_h
    if abs(float(transformed[3])) < 1e-12:
        raise ValueError("Transformed homogeneous point has near-zero w.")
    return transformed[:3] / transformed[3]


def main() -> None:
    args = _parse_args()

    t_base_tcp, tcp_source = _resolve_base_tcp(args)
    t_tcp_camera = _load_transform(args.handeye_file, "T_tcp_camera")
    p_camera_tag, tag_source = _resolve_camera_tag_point(args)

    p_base_tag = _transform_point(t_base_tcp @ t_tcp_camera, p_camera_tag)

    if args.quiet:
        print(f"{p_base_tag[0]:.9f} {p_base_tag[1]:.9f} {p_base_tag[2]:.9f}")
        return

    print("\nEye-on-Hand tag center in robot base frame")
    print("=" * 58)
    print(f"TCP source: {tcp_source}")
    print(f"Hand-eye file: {args.handeye_file}")
    print(f"Tag camera source: {tag_source}")
    print("\nInput tag center in camera frame (m):")
    print(f"  x={p_camera_tag[0]: .9f}, y={p_camera_tag[1]: .9f}, z={p_camera_tag[2]: .9f}")
    print("\nOutput tag center in base frame:")
    print(f"  meters: x={p_base_tag[0]: .9f}, y={p_base_tag[1]: .9f}, z={p_base_tag[2]: .9f}")
    print(
        "  millimeters: "
        f"x={p_base_tag[0] * 1000.0: .3f}, "
        f"y={p_base_tag[1] * 1000.0: .3f}, "
        f"z={p_base_tag[2] * 1000.0: .3f}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
