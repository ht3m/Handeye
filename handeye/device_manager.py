#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
设备管理器 - 负责连接和管理 UR5 机械臂与 RealSense 相机
"""

import numpy as np
from typing import Optional, cast

# 添加项目根目录到路径
from handeye.robot.ur_robot import URRobot
from handeye.camera.realsense import RealSenseCamera
from handeye.config import UR5_CONFIG, REALSENSE_CONFIG


class DeviceManager:
    """设备管理器 - 统一管理机械臂和相机连接"""

    def __init__(self) -> None:
        self.robot: Optional[URRobot] = None
        self.camera: Optional[RealSenseCamera] = None
        self.connected = False

    def connect(
        self,
        robot_ip: Optional[str] = None,
        connect_robot: bool = True,
        connect_camera: bool = True
    ) -> bool:
        """
        连接设备

        Args:
            robot_ip: UR5 IP地址
            connect_robot: 是否连接机械臂
            connect_camera: 是否连接相机

        Returns:
            success: 连接是否成功
        """
        print("=" * 50)
        print("设备连接")
        print("=" * 50)

        success = True

        # 连接机械臂
        if connect_robot:
            if robot_ip is None:
                robot_ip = cast(str, UR5_CONFIG['tcp_host_ip'])
            tcp_port = int(cast(int, UR5_CONFIG['tcp_port']))
            print(f"\n[1/2] 连接 UR5 机械臂: {robot_ip} ...")
            try:
                self.robot = URRobot(robot_ip, tcp_port)
                # 测试连接
                pose = self.robot.get_tool_pose()
                if pose is not None:
                    print(f"  ✓ 机械臂连接成功")
                    print(f"  当前TCP位姿: {np.array(pose[:3]).round(4)}")
                else:
                    print(f"  ✗ 无法获取机械臂位姿")
                    success = False
            except Exception as e:
                print(f"  ✗ 机械臂连接失败: {e}")
                success = False
        else:
            print("\n[1/2] 跳过机械臂连接")

        # 连接相机
        if connect_camera:
            print("\n[2/2] 连接 RealSense D405 相机 ...")
            try:
                self.camera = RealSenseCamera()
                self.camera.connect()
                intrinsics = self.camera.intrinsics
                if intrinsics is not None:
                    print(f"  ✓ 相机连接成功")
                    print(f"  分辨率: {REALSENSE_CONFIG['width']}x{REALSENSE_CONFIG['height']}")
                else:
                    print(f"  ✗ 无法获取相机内参")
                    success = False
            except Exception as e:
                print(f"  ✗ 相机连接失败: {e}")
                success = False
        else:
            print("\n[2/2] 跳过相机连接")

        self.connected = success
        return success

    def disconnect(self) -> None:
        """断开设备连接"""
        if self.camera is not None:
            try:
                self.camera.disconnect()
                print("相机已断开")
            except:
                pass

        if self.robot is not None:
            try:
                self.robot.disconnect()
                print("机械臂已断开")
            except:
                pass

        self.connected = False

    def get_robot(self) -> Optional[URRobot]:
        """获取机械臂对象"""
        return self.robot

    def get_camera(self) -> Optional[RealSenseCamera]:
        """获取相机对象"""
        return self.camera

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.connected
