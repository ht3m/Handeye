#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UR5 机械臂通信接口
参考 RoboVision/Tools/position.py 的读取方式：
直接从 RTDE 数据 offset 444 读取 6 个 double 获取 TCP 位姿。
"""

import socket
import struct
import time
import numpy as np
from typing import Optional


class URRobot:
    """UR5 机械臂控制类 - 轻量级位姿读取"""

    def __init__(
        self,
        tcp_host_ip: str = "169.254.162.96",
        tcp_port: int = 30003,
        workspace_limits: Optional[np.ndarray] = None
    ) -> None:
        """
        初始化 UR5 机械臂连接

        Args:
            tcp_host_ip: UR5 IP地址
            tcp_port: TCP端口号 (默认30003)
            workspace_limits: 工作空间限制
        """
        self.tcp_host_ip = tcp_host_ip
        self.tcp_port = tcp_port
        self.tcp_socket: Optional[socket.socket] = None

        # 工作空间限制
        if workspace_limits is None:
            workspace_limits_arr = np.array(
                [[-0.7, 0.7], [-0.7, 0.7], [0.00, 0.6]], dtype=np.float64
            )
        else:
            workspace_limits_arr = np.asarray(workspace_limits, dtype=np.float64)
        self.workspace_limits: np.ndarray = workspace_limits_arr

    def connect(self) -> None:
        """建立 TCP 连接"""
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        assert self.tcp_socket is not None
        self.tcp_socket.settimeout(5.0)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        print(f"已连接到 UR5: {self.tcp_host_ip}:{self.tcp_port}")

    def disconnect(self) -> None:
        """断开 TCP 连接"""
        if self.tcp_socket:
            self.tcp_socket.close()
            self.tcp_socket = None

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.tcp_socket is not None

    # ==================== 位姿获取 ====================

    def get_tool_pose(self) -> np.ndarray:
        """
        获取当前末端位姿 (TCP)
        参考 position.py: 从 RTDE 数据 offset 444 读取 6 个 double

        Returns:
            np.array: [x, y, z, rx, ry, rz] (位置: 米, 旋转向量: 弧度)
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((self.tcp_host_ip, self.tcp_port))
            data = sock.recv(1108)
            sock.close()

            # offset 444 → 6 个双精度浮点数 = TCP 位姿
            pose = np.array(
                struct.unpack('!6d', data[444:444 + 48]),
                dtype=np.float64
            )
            return pose

        except Exception as e:
            raise RuntimeError(f"UR5 位姿读取失败: {e}")

    def get_transform_matrix(self) -> np.ndarray:
        """
        获取末端齐次变换矩阵 (4x4)

        Returns:
            np.array: 4x4 齐次变换矩阵
        """
        tool_pose = self.get_tool_pose()
        return self.pose_to_transform(tool_pose)

    # ==================== 坐标变换 ====================

    @staticmethod
    def rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
        """旋转向量转旋转矩阵 (Rodrigues formula)"""
        theta = float(np.linalg.norm(rotvec))
        if theta < 1e-8:
            return np.eye(3, dtype=np.float64)
        k = rotvec / theta
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ], dtype=np.float64)
        R = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R

    @staticmethod
    def R_to_rotvec(R: np.ndarray) -> np.ndarray:
        """旋转矩阵转旋转向量"""
        theta = float(np.arccos((np.trace(R) - 1.0) / 2.0))
        if abs(theta) < 1e-8:
            return np.zeros(3, dtype=np.float64)
        rx = (R[2, 1] - R[1, 2]) / (2.0 * np.sin(theta))
        ry = (R[0, 2] - R[2, 0]) / (2.0 * np.sin(theta))
        rz = (R[1, 0] - R[0, 1]) / (2.0 * np.sin(theta))
        return np.array([rx, ry, rz], dtype=np.float64) * theta

    @staticmethod
    def pose_to_transform(pose: np.ndarray) -> np.ndarray:
        """
        位姿数组转齐次变换矩阵

        Args:
            pose: [x, y, z, rx, ry, rz]

        Returns:
            np.array: 4x4 齐次变换矩阵
        """
        x, y, z, rx, ry, rz = pose
        R = URRobot.rotvec_to_R(np.array([rx, ry, rz], dtype=np.float64))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def transform_to_pose(T: np.ndarray) -> np.ndarray:
        """
        齐次变换矩阵转位姿数组

        Args:
            T: 4x4 齐次变换矩阵

        Returns:
            np.array: [x, y, z, rx, ry, rz]
        """
        pos = T[:3, 3]
        rotvec = URRobot.R_to_rotvec(T[:3, :3])
        return np.concatenate([pos, rotvec])

    # ==================== 工作空间检查 ====================

    def is_in_workspace(self, pose: np.ndarray) -> bool:
        """检查位姿是否在工作空间内"""
        pos = pose[:3]
        return bool(
            self.workspace_limits[0, 0] <= pos[0] <= self.workspace_limits[0, 1]
            and self.workspace_limits[1, 0] <= pos[1] <= self.workspace_limits[1, 1]
            and self.workspace_limits[2, 0] <= pos[2] <= self.workspace_limits[2, 1]
        )


if __name__ == "__main__":
    # 测试代码
    robot = URRobot(tcp_host_ip="169.254.162.96")

    print("获取当前位姿...")
    pose = robot.get_tool_pose()
    print(f"当前末端位姿: {pose}")

    T = robot.get_transform_matrix()
    print("末端变换矩阵:")
    print(T)