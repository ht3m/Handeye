#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize the relative pose between robot TCP and camera optical frame.

This tool is intended for a quick sanity check before numerical error
evaluation.  It draws the robot end-effector frame and the camera optical frame
in one local coordinate system, using the saved hand-eye transform.
"""

import argparse
import os
import sys
from typing import Any, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from handeye.calibration.transforms import invert_transform  # noqa: E402
from handeye.config import CALIBRATION_MODE, get_results_path  # noqa: E402
from handeye.transform_io import load_transform_matrix  # noqa: E402


AXIS_COLORS = ("red", "green", "blue")
AXIS_NAMES = ("X", "Y", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize robot TCP frame and camera optical frame from a saved "
            "hand-eye transform."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("eye_on_hand", "eye_to_hand"),
        default=CALIBRATION_MODE,
        help="Calibration mode used to choose the default result path.",
    )
    parser.add_argument(
        "--transform",
        default=None,
        help=(
            "Path to a 4x4 transform txt file. Defaults to "
            "results/<mode>/handeye_transform.txt."
        ),
    )
    parser.add_argument(
        "--inverse",
        action="store_true",
        help="Invert the loaded transform before drawing it.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.06,
        help="Coordinate axis length in meters.",
    )
    parser.add_argument(
        "--frustum-depth",
        type=float,
        default=0.08,
        help="Camera frustum depth in meters.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional image path for saving the figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create the figure without opening an interactive window.",
    )
    return parser.parse_args()


def load_transform(path: str) -> np.ndarray:
    return load_transform_matrix(path)


def rotation_to_xyz_euler_deg(R: np.ndarray) -> np.ndarray:
    """Return intrinsic XYZ Euler angles in degrees for concise display."""
    sy = float(np.clip(R[0, 2], -1.0, 1.0))
    y = np.arcsin(sy)
    cy = np.cos(y)

    if abs(cy) > 1e-8:
        x = np.arctan2(-R[1, 2], R[2, 2])
        z = np.arctan2(-R[0, 1], R[0, 0])
    else:
        x = np.arctan2(R[2, 1], R[1, 1])
        z = 0.0

    return np.degrees(np.array([x, y, z], dtype=np.float64))


def draw_frame(
    ax: Any,
    T: np.ndarray,
    name: str,
    axis_length: float,
    origin_color: str,
) -> None:
    origin = T[:3, 3]
    R = T[:3, :3]
    ax.scatter(*origin, color=origin_color, s=80)
    ax.text(*origin, f" {name}", color=origin_color, fontsize=10, weight="bold")

    for i, (axis_name, color) in enumerate(zip(AXIS_NAMES, AXIS_COLORS)):
        direction = R[:, i] * axis_length
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=color,
            arrow_length_ratio=0.16,
            linewidth=2.2,
        )
        end = origin + direction * 1.08
        ax.text(*end, f"{name}-{axis_name}", color=color, fontsize=8)


def draw_camera_frustum(ax: Any, T_tcp_camera: np.ndarray, depth: float) -> None:
    origin = T_tcp_camera[:3, 3]
    R = T_tcp_camera[:3, :3]
    half_w = depth * 0.45
    half_h = depth * 0.32

    corners_cam = np.array(
        [
            [-half_w, -half_h, depth],
            [half_w, -half_h, depth],
            [half_w, half_h, depth],
            [-half_w, half_h, depth],
        ],
        dtype=np.float64,
    )
    corners_tcp = (R @ corners_cam.T).T + origin

    for corner in corners_tcp:
        ax.plot(
            [origin[0], corner[0]],
            [origin[1], corner[1]],
            [origin[2], corner[2]],
            color="0.35",
            linewidth=1.0,
            alpha=0.8,
        )

    closed = np.vstack([corners_tcp, corners_tcp[0]])
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="0.25", linewidth=1.2)


def collect_points(transforms: Iterable[np.ndarray], extra: Iterable[np.ndarray]) -> np.ndarray:
    points = [np.asarray(T[:3, 3], dtype=np.float64) for T in transforms]
    points.extend(np.asarray(p, dtype=np.float64).reshape(3) for p in extra)
    return np.vstack(points)


def set_equal_axes(ax: Any, points: np.ndarray, margin: float) -> None:
    center = points.mean(axis=0)
    span = np.ptp(points, axis=0)
    radius = max(float(span.max()) * 0.6 + margin, margin)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def attach_esc_close(fig: Any) -> None:
    def on_key(event: Any) -> None:
        if event is not None and event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)


def print_summary(T_tcp_camera: np.ndarray, source_path: str, inverted: bool) -> None:
    R = T_tcp_camera[:3, :3]
    t = T_tcp_camera[:3, 3]
    euler = rotation_to_xyz_euler_deg(R)
    orthogonality_error = float(np.linalg.norm(R.T @ R - np.eye(3)))
    determinant = float(np.linalg.det(R))

    print("\nHand-eye frame sanity check")
    print("=" * 50)
    print(f"Transform file: {source_path}")
    print(f"Inverted before drawing: {inverted}")
    print("Interpreted transform: T_tcp_camera")
    print(f"Camera origin in TCP frame (m): x={t[0]: .4f}, y={t[1]: .4f}, z={t[2]: .4f}")
    print(f"Camera origin in TCP frame (mm): x={t[0] * 1000: .1f}, y={t[1] * 1000: .1f}, z={t[2] * 1000: .1f}")
    print(f"Camera XYZ Euler relative to TCP (deg): rx={euler[0]: .2f}, ry={euler[1]: .2f}, rz={euler[2]: .2f}")
    print(f"Rotation determinant: {determinant:.6f}")
    print(f"Rotation orthogonality error: {orthogonality_error:.3e}")
    print("\nVisual check points:")
    print("- Camera-Z (blue) is the optical forward direction.")
    print("- Camera-X (red) points to image right in the usual OpenCV optical frame.")
    print("- Camera-Y (green) points to image down in the usual OpenCV optical frame.")
    print("- Compare the camera origin offset and optical axis direction with the real mount.")


def build_plot(
    T_tcp_camera: np.ndarray,
    axis_length: float,
    frustum_depth: float,
) -> Tuple[Any, Any]:
    tcp_T = np.eye(4, dtype=np.float64)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    draw_frame(ax, tcp_T, "TCP", axis_length, "black")
    draw_frame(ax, T_tcp_camera, "Camera", axis_length, "darkorange")
    draw_camera_frustum(ax, T_tcp_camera, frustum_depth)

    cam_origin = T_tcp_camera[:3, 3]
    ax.plot(
        [0.0, cam_origin[0]],
        [0.0, cam_origin[1]],
        [0.0, cam_origin[2]],
        color="purple",
        linestyle="--",
        linewidth=1.5,
        label="TCP to camera origin",
    )

    points = collect_points(
        [tcp_T, T_tcp_camera],
        [T_tcp_camera[:3, 3] + T_tcp_camera[:3, 2] * frustum_depth],
    )
    set_equal_axes(ax, points, margin=max(axis_length, frustum_depth) * 1.6)

    ax.set_xlabel("TCP X (m)")
    ax.set_ylabel("TCP Y (m)")
    ax.set_zlabel("TCP Z (m)")
    ax.set_title("Hand-eye Result: TCP Frame and Camera Optical Frame")
    ax.legend(
        handles=[
            Line2D([0], [0], color="red", lw=2, label="Frame X-axis"),
            Line2D([0], [0], color="green", lw=2, label="Frame Y-axis"),
            Line2D([0], [0], color="blue", lw=2, label="Frame Z-axis"),
            Line2D([0], [0], color="purple", lw=1.5, linestyle="--", label="TCP to camera"),
            Line2D([0], [0], color="0.25", lw=1.2, label="Camera frustum"),
        ],
        loc="upper left",
    )
    attach_esc_close(fig)
    plt.tight_layout()
    return fig, ax


def main() -> None:
    args = parse_args()
    transform_path = args.transform or os.path.join(
        get_results_path(args.mode),
        "handeye_transform.txt",
    )

    T_tcp_camera = load_transform(transform_path)
    if args.inverse:
        T_tcp_camera = invert_transform(T_tcp_camera)

    print_summary(T_tcp_camera, transform_path, args.inverse)
    fig, _ax = build_plot(T_tcp_camera, args.axis_length, args.frustum_depth)

    if args.save:
        fig.savefig(args.save, dpi=180, bbox_inches="tight")
        print(f"\nSaved figure: {args.save}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nVisualization interrupted.")
