#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Result visualization module - Unified coordinate system for TCP, camera, calibration board display
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from typing import Any, List, Optional

from handeye.calibration.transforms import pose_to_mat


class ResultVisualizer:
    """Calibration result visualizer - Unified to UR5 base coordinate system"""

    def __init__(self, mode: str) -> None:
        """
        Initialize visualizer

        Args:
            mode: 'eye_on_hand' or 'eye_to_hand'
        """
        self.mode = mode

    @staticmethod
    def _attach_esc_close(fig: Any) -> None:
        """Bind Esc key to close current matplotlib figure."""

        def on_key(event: Any) -> None:
            if event is not None and event.key == 'escape':
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)

    def visualize(
        self,
        robot_poses: List[np.ndarray],
        X: np.ndarray,
        z_scale: float = 1.0,
        frame_scale: float = 1.0,
        camera_poses: Optional[List[np.ndarray]] = None,
        board_to_base: Optional[List[float]] = None,
        board_to_tcp: Optional[List[float]] = None
    ) -> None:
        """
        Visualize calibration result

        Args:
            robot_poses: TCP pose list (in base coordinate system)
            X: hand-eye transformation matrix (camera relative to TCP or base)
            z_scale: depth scale factor
            frame_scale: coordinate frame axis length scale factor
            camera_poses: board pose list in camera frame (T_cam_board)
            board_to_base: rough pose of calibration board relative to base (for Eye-on-Hand)
            board_to_tcp: rough pose of calibration board relative to TCP (for Eye-to-Hand)
        """
        print("\nGenerating visualization...")
        frame_scale = max(0.1, float(frame_scale))

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 1. Draw all TCP positions (blue dots)
        tcp_positions = np.array([p[:3, 3] for p in robot_poses])
        ax.scatter(tcp_positions[:, 0], tcp_positions[:, 1], tcp_positions[:, 2],
                  c='blue', marker='o', s=80, label='TCP (flange)', alpha=0.7)
        if robot_poses:
            # Show one explicit TCP coordinate frame for readability.
            self._draw_coordinate_frame(ax, robot_poses[0], 'TCP', 'blue', 0.08 * frame_scale)

        # 2. Calculate and draw camera positions based on mode
        if self.mode == 'eye_on_hand':
            # Eye-on-Hand: Camera moves with TCP
            # Camera in base coordinate = TCP @ X
            camera_positions: List[np.ndarray] = []
            for tcp in robot_poses:
                T_cam_in_base = tcp @ X
                camera_positions.append(T_cam_in_base[:3, 3])
            camera_positions_np = np.array(camera_positions)
            ax.scatter(camera_positions_np[:, 0], camera_positions_np[:, 1], camera_positions_np[:, 2],
                      c='red', marker='^', s=60, label='Camera (Eye-on-Hand)', alpha=0.7)

            # Draw camera coordinate frame (using first position orientation)
            self._draw_coordinate_frame(ax, robot_poses[0] @ X, 'Camera', 'red', 0.05 * frame_scale)

        else:
            # Eye-to-Hand: Camera is fixed
            # Camera in base coordinate = X (directly from calibration result)
            cam_pos = X[:3, 3]
            ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
                      c='red', marker='^', s=150, label='Camera (Eye-to-Hand)', alpha=0.9)

            # Draw camera coordinate frame
            self._draw_coordinate_frame(ax, X, 'Camera', 'red', 0.1 * frame_scale)

        # 3. Draw calibration board positions (using rough pose as reference)
        if self.mode == 'eye_on_hand' and board_to_base is not None:
            # Calibration board fixed on base
            T_board = pose_to_mat(board_to_base)
            self._draw_coordinate_frame(ax, T_board, 'Board (ref)', 'green', 0.1 * frame_scale)

            # Draw measured board positions from observations
            if camera_poses is not None:
                for tcp, cam_pose in zip(robot_poses, camera_poses):
                    T_board_measured = tcp @ X @ cam_pose
                    p = T_board_measured[:3, 3]
                    ax.scatter([p[0]], [p[1]], [p[2]],
                              c='green', marker='s', s=30, alpha=0.4)

                # Show one estimated board coordinate frame.
                if robot_poses and camera_poses:
                    T_board_est = robot_poses[0] @ X @ camera_poses[0]
                    self._draw_coordinate_frame(ax, T_board_est, 'Board (est)', 'lime', 0.08 * frame_scale)

        elif self.mode == 'eye_to_hand' and board_to_tcp is not None:
            # Calibration board moves with TCP (reference trajectory)
            T_board_rough = pose_to_mat(board_to_tcp)
            for tcp in robot_poses:
                T_board_ref = tcp @ T_board_rough
                ax.scatter([T_board_ref[0, 3]], [T_board_ref[1, 3]], [T_board_ref[2, 3]],
                          c='green', marker='s', s=30, alpha=0.4)

            # Draw a sample board coordinate frame
            if robot_poses:
                self._draw_coordinate_frame(ax, robot_poses[0] @ T_board_rough,
                                          'Board (ref)', 'green', 0.1 * frame_scale)

            # Draw measured board positions from observations
            if camera_poses is not None:
                for cam_pose in camera_poses:
                    T_board_measured = X @ cam_pose
                    p = T_board_measured[:3, 3]
                    ax.scatter([p[0]], [p[1]], [p[2]],
                              c='lime', marker='x', s=25, alpha=0.5)

                # Show one estimated board coordinate frame.
                if camera_poses:
                    T_board_est = X @ camera_poses[0]
                    self._draw_coordinate_frame(ax, T_board_est, 'Board (est)', 'lime', 0.08 * frame_scale)

        # 4. Draw base coordinate frame
        self._draw_coordinate_frame(ax, np.eye(4), 'Base', 'black', 0.15 * frame_scale)

        # Set figure properties
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)

        mode_name = "Eye-on-Hand" if self.mode == "eye_on_hand" else "Eye-to-Hand"
        ax.set_title(f'{mode_name} Calibration Result\n(Unified coordinate: UR5 Base)', fontsize=14)

        # Build explicit legend for points and axis-color meanings.
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8,
                   label='TCP (flange)'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8,
                   label='Camera'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=8,
                   label='Board (reference)'),
            Line2D([0], [0], marker='x', color='lime', markersize=8,
                   label='Board (estimated)'),
            Line2D([0], [0], color='r', lw=2, label='Frame X-axis'),
            Line2D([0], [0], color='g', lw=2, label='Frame Y-axis'),
            Line2D([0], [0], color='b', lw=2, label='Frame Z-axis'),
        ]
        ax.legend(handles=legend_handles, loc='upper left', fontsize=10)

        # Set equal axis scale
        self._set_equal_axis(ax, tcp_positions)

        # Enable mouse-wheel zoom on 3D axes.
        self._enable_scroll_zoom(fig, ax)
        self._attach_esc_close(fig)

        plt.tight_layout()
        plt.show()

    def _enable_scroll_zoom(self, fig: Any, ax: Any, zoom_step: float = 1.2) -> None:
        """Enable mouse wheel zoom for matplotlib 3D axes."""

        def on_scroll(event: Any) -> None:
            if event.inaxes != ax:
                return

            if event.button == 'up':
                scale = 1.0 / zoom_step
            elif event.button == 'down':
                scale = zoom_step
            else:
                return

            xlim = ax.get_xlim3d()
            ylim = ax.get_ylim3d()
            zlim = ax.get_zlim3d()

            xmid = 0.5 * (xlim[0] + xlim[1])
            ymid = 0.5 * (ylim[0] + ylim[1])
            zmid = 0.5 * (zlim[0] + zlim[1])

            xhalf = 0.5 * (xlim[1] - xlim[0]) * scale
            yhalf = 0.5 * (ylim[1] - ylim[0]) * scale
            zhalf = 0.5 * (zlim[1] - zlim[0]) * scale

            ax.set_xlim3d([xmid - xhalf, xmid + xhalf])
            ax.set_ylim3d([ymid - yhalf, ymid + yhalf])
            ax.set_zlim3d([zmid - zhalf, zmid + zhalf])
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('scroll_event', on_scroll)

    def visualize_errors(
        self,
        errors: np.ndarray,
        error_name: str = 'Position Error',
        unit: str = 'mm',
        scale: float = 1000.0
    ) -> None:
        """
        Visualize error distribution

        Args:
            errors: error array
            error_name: error type name shown in plots
            unit: displayed unit
            scale: value scale from raw errors to displayed unit
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        displayed = errors * scale
        mean_v = float(np.mean(displayed)) if len(displayed) > 0 else 0.0

        # Error bar chart
        ax1 = axes[0]
        ax1.bar(range(1, len(displayed)+1), displayed, color='steelblue', edgecolor='black')
        ax1.axhline(y=mean_v, color='r', linestyle='--',
                   label=f'Mean: {mean_v:.3f} {unit}')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel(f'Error ({unit})')
        ax1.set_title(f'{error_name} per Frame')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Error histogram
        ax2 = axes[1]
        ax2.hist(displayed, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=mean_v, color='r', linestyle='--',
                   label=f'Mean: {mean_v:.3f} {unit}')
        ax2.set_xlabel(f'Error ({unit})')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{error_name} Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'{error_name} Analysis', fontsize=14)
        self._attach_esc_close(fig)
        plt.tight_layout()
        plt.show()

    def visualize_position_rotation_errors(
        self,
        position_errors: np.ndarray,
        rotation_errors: np.ndarray,
        pos_unit: str = 'mm',
        rot_unit: str = 'deg',
        pos_scale: float = 1000.0,
        rot_scale: float = 1.0
    ) -> None:
        """
        Simultaneously visualize position and rotation error distributions.

        Args:
            position_errors: position error array
            rotation_errors: rotation error array
            pos_unit: displayed unit for position
            rot_unit: displayed unit for rotation
            pos_scale: scale from raw position values to displayed unit
            rot_scale: scale from raw rotation values to displayed unit
        """
        pos_display = position_errors * pos_scale
        rot_display = rotation_errors * rot_scale

        pos_mean = float(np.mean(pos_display)) if len(pos_display) > 0 else 0.0
        rot_mean = float(np.mean(rot_display)) if len(rot_display) > 0 else 0.0

        fig, axes = plt.subplots(2, 2, figsize=(15, 9))

        # Position error per frame
        ax = axes[0, 0]
        ax.bar(range(1, len(pos_display) + 1), pos_display, color='steelblue', edgecolor='black')
        ax.axhline(y=pos_mean, color='r', linestyle='--', label=f'Mean: {pos_mean:.3f} {pos_unit}')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel(f'Error ({pos_unit})')
        ax.set_title('Position Error per Frame')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Position error histogram
        ax = axes[0, 1]
        ax.hist(pos_display, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=pos_mean, color='r', linestyle='--', label=f'Mean: {pos_mean:.3f} {pos_unit}')
        ax.set_xlabel(f'Error ({pos_unit})')
        ax.set_ylabel('Frequency')
        ax.set_title('Position Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rotation error per frame
        ax = axes[1, 0]
        ax.bar(range(1, len(rot_display) + 1), rot_display, color='darkorange', edgecolor='black')
        ax.axhline(y=rot_mean, color='r', linestyle='--', label=f'Mean: {rot_mean:.3f} {rot_unit}')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel(f'Error ({rot_unit})')
        ax.set_title('Rotation Error per Frame')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rotation error histogram
        ax = axes[1, 1]
        ax.hist(rot_display, bins=15, color='darkorange', edgecolor='black', alpha=0.7)
        ax.axvline(x=rot_mean, color='r', linestyle='--', label=f'Mean: {rot_mean:.3f} {rot_unit}')
        ax.set_xlabel(f'Error ({rot_unit})')
        ax.set_ylabel('Frequency')
        ax.set_title('Rotation Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Position & Rotation Error Analysis', fontsize=14)
        self._attach_esc_close(fig)
        plt.tight_layout()
        plt.show()

    def _draw_coordinate_frame(
        self,
        ax: Any,
        T: np.ndarray,
        label: str,
        color: str,
        length: float = 0.1
    ) -> None:
        """
        Draw coordinate frame

        Args:
            ax: matplotlib 3D axis
            T: 4x4 transformation matrix
            label: label
            color: color ('red', 'green', 'blue', 'black')
            length: axis length
        """
        pos = T[:3, 3]
        R = T[:3, :3]

        # Calculate each axis endpoint
        x_axis = pos + R[:, 0] * length
        y_axis = pos + R[:, 1] * length
        z_axis = pos + R[:, 2] * length

        # Draw origin point
        ax.scatter([pos[0]], [pos[1]], [pos[2]], color=color, s=100)

        # Draw axes
        ax.quiver(pos[0], pos[1], pos[2],
                 x_axis[0]-pos[0], x_axis[1]-pos[1], x_axis[2]-pos[2],
                 color='r', arrow_length_ratio=0.1, linewidth=2)
        ax.quiver(pos[0], pos[1], pos[2],
                 y_axis[0]-pos[0], y_axis[1]-pos[1], y_axis[2]-pos[2],
                 color='g', arrow_length_ratio=0.1, linewidth=2)
        ax.quiver(pos[0], pos[1], pos[2],
                 z_axis[0]-pos[0], z_axis[1]-pos[1], z_axis[2]-pos[2],
                 color='b', arrow_length_ratio=0.1, linewidth=2)

        # Label
        ax.text(pos[0], pos[1], pos[2], label, color=color, fontsize=10,
               weight='bold')

    def _set_equal_axis(self, ax: Any, *positions: np.ndarray) -> None:
        """Set equal axis scale"""
        all_points = []
        for pos_array in positions:
            if pos_array is None:
                continue

            arr = np.asarray(pos_array)
            if arr.size == 0:
                continue

            # Accept either point arrays (N,3) or pose arrays (N,4,4)/(4,4).
            if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                arr = arr[:, :3, 3]
            elif arr.ndim == 2 and arr.shape == (4, 4):
                arr = arr[:3, 3].reshape(1, 3)
            elif arr.ndim == 1 and arr.shape[0] == 3:
                arr = arr.reshape(1, 3)

            if arr.ndim == 2 and arr.shape[1] == 3:
                all_points.append(arr)

        if not all_points:
            return

        all_points_arr = np.vstack(all_points)
        max_range = float(np.abs(all_points_arr).max()) * 1.2

        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
