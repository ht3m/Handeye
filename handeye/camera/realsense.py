#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Intel RealSense D405 相机驱动
支持RGB-D图像采集、内参获取
"""

import json
import os
import time
import numpy as np
import pyrealsense2 as rs
import cv2
from typing import Any, Dict, Optional, Tuple, cast


class RealSenseCamera:
    """RealSense D405 相机类"""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        device_id: Optional[str] = None
    ) -> None:
        """
        初始化RealSense相机

        Args:
            width: 图像宽度
            height: 图像高度
            fps: 帧率
            device_id: 设备ID (可选)
        """
        self.im_width = width
        self.im_height = height
        self.fps = fps
        self.device_id = device_id

        self.pipeline: Optional[Any] = None
        self.config: Optional[Any] = None
        self.intrinsics: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.depth_scale: Optional[float] = None
        self.align: Optional[Any] = None

        self.connected = False

    def connect(self) -> None:
        """连接并启动相机"""
        # 创建管道
        self.pipeline = rs.pipeline()

        # 配置流
        self.config = rs.config()
        config = self.config
        if self.device_id:
            config.enable_device(self.device_id)

        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.fps)

        # 启动管道
        pipeline = self.pipeline
        profile = pipeline.start(config)

        # 获取内参
        rgb_profile = profile.get_stream(rs.stream.color)
        self.intrinsics = np.asarray(self._get_intrinsics(rgb_profile), dtype=np.float64)
        self.dist_coeffs = np.asarray(self._get_dist_coeffs(rgb_profile), dtype=np.float64).reshape(-1, 1)

        # 获取深度缩放因子
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

        # 创建对齐对象 (对齐到彩色图)
        self.align = rs.align(rs.stream.color)

        self.connected = True
        print(f"RealSense D405 已连接")
        print(f"  分辨率: {self.im_width}x{self.im_height}")
        print(f"  内参:\n{self.intrinsics}")
        print(f"  畸变参数: {self.dist_coeffs.ravel() if self.dist_coeffs is not None else 'None'}")
        print(f"  深度缩放: {self.depth_scale}")

    def disconnect(self) -> None:
        """断开相机连接"""
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        self.connected = False
        print("RealSense D405 已断开")

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取RGB-D图像

        Returns:
            color_img: BGR格式彩色图 (H, W, 3)
            depth_img: 深度图 (H, W), 单位: 米
        """
        if not self.connected:
            self.connect()
        pipeline = self.pipeline
        align = self.align
        depth_scale = self.depth_scale
        assert pipeline is not None and align is not None and depth_scale is not None
        # 等待帧
        frames = pipeline.wait_for_frames()

        # 对齐深度图到彩色图
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 转换为numpy数组
        # 深度图: uint16 -> 米
        depth_image = np.asarray(depth_frame.get_data(), dtype=np.float32)
        depth_image = depth_image * depth_scale  # 转换为米

        # 彩色图: BGR格式
        color_image = np.asarray(color_frame.get_data(), dtype=np.uint8)

        return color_image, depth_image

    def get_color_image(self) -> np.ndarray:
        """获取彩色图像"""
        if not self.connected:
            self.connect()
        pipeline = self.pipeline
        assert pipeline is not None
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        return np.asarray(color_frame.get_data(), dtype=np.uint8)

    def get_depth_image(self) -> np.ndarray:
        """获取深度图像 (单位: 米)"""
        if not self.connected:
            self.connect()
        pipeline = self.pipeline
        align = self.align
        depth_scale = self.depth_scale
        assert pipeline is not None and align is not None and depth_scale is not None

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asarray(depth_frame.get_data(), dtype=np.float32)
        depth_image = depth_image * depth_scale
        return depth_image

    def get_aligned_depth(self, u: int, v: int) -> float:
        """
        获取指定像素点的深度值

        Args:
            u, v: 像素坐标

        Returns:
            depth: 深度值 (米)
        """
        if not self.connected:
            self.connect()

        depth_img = self.get_depth_image()
        if 0 <= v < depth_img.shape[0] and 0 <= u < depth_img.shape[1]:
            return float(depth_img[v, u])
        return 0.0

    def project_to_3d(self, u: int, v: int, depth: float) -> np.ndarray:
        """
        像素坐标转相机坐标系3D点

        Args:
            u, v: 像素坐标
            depth: 深度值 (米)

        Returns:
            point_3d: [X, Y, Z] 相机坐标系下的3D点
        """
        intrinsics = self.intrinsics
        assert intrinsics is not None
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        return np.array([X, Y, Z], dtype=np.float64)

    def _get_intrinsics(self, rgb_profile: Any) -> np.ndarray:
        """
        获取相机内参

        Args:
            rgb_profile: RGB流配置

        Returns:
            intrinsics: 3x3 内参矩阵
        """
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # 构建内参矩阵
        # [[fx, 0, cx],
        #  [0, fy, cy],
        #  [0, 0, 1]]
        intrinsics = np.array([
            [raw_intrinsics.fx, 0, raw_intrinsics.ppx],
            [0, raw_intrinsics.fy, raw_intrinsics.ppy],
            [0, 0, 1]
        ])

        return intrinsics

    def _get_dist_coeffs(self, rgb_profile: Any) -> np.ndarray:
        """获取相机畸变系数 (k1, k2, p1, p2, k3)."""
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        coeffs = list(raw_intrinsics.coeffs)
        if len(coeffs) >= 5:
            return np.array(coeffs[:5], dtype=np.float64)
        return np.zeros(5, dtype=np.float64)

    def get_default_intrinsics(self) -> np.ndarray:
        """获取D405默认内参 (可用于未标定的情况)"""
        return np.array([
            [615.284, 0, 309.623],
            [0, 614.557, 247.967],
            [0, 0, 1]
        ])

    def display(self, wait_key: int = 1) -> None:
        """
        显示RGB-D图像

        Args:
            wait_key: cv2.waitKey 参数
        """
        color_img, depth_img = self.get_data()

        # 深度图可视化
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img * 1000, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # 合并显示
        if depth_img.shape[:2] != color_img.shape[:2]:
            # 分辨率不同时调整大小
            depth_vis = cv2.resize(depth_vis, (color_img.shape[1], color_img.shape[0]))

        combined = np.hstack([color_img, depth_vis])
        cv2.imshow('RealSense D405 - Color | Depth', combined)
        cv2.waitKey(wait_key)

    def close(self) -> None:
        """关闭显示窗口"""
        cv2.destroyAllWindows()

    def write_intrinsics_snapshot(self, output_root: str = "data/svd") -> str:
        """Write one camera intrinsics snapshot for offline experiments."""
        if not self.connected:
            self.connect()

        intrinsics = self.intrinsics
        dist_coeffs = self.dist_coeffs
        depth_scale = self.depth_scale
        assert intrinsics is not None and dist_coeffs is not None and depth_scale is not None

        images_dir = os.path.join(output_root, "images")
        os.makedirs(images_dir, exist_ok=True)
        snapshot_path = os.path.join(images_dir, "camera_intrinsics.json")

        payload: Dict[str, Any] = {
            "image_width": int(self.im_width),
            "image_height": int(self.im_height),
            "fps": int(self.fps),
            "depth_scale": float(depth_scale),
            "intrinsics": intrinsics.tolist(),
            "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
            "captured_at_ns": int(time.time_ns()),
        }

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return snapshot_path

    def save_rgbd_frame(
        self,
        frame_index: int,
        output_root: str = "data/svd",
        color_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        timestamp_ns: Optional[int] = None,
    ) -> Dict[str, str]:
        """Save one RGB-D frame and append timestamp for offline analysis.

        The color image is saved in OpenCV BGR channel order.
        """
        if frame_index <= 0:
            raise ValueError("frame_index must be >= 1")

        images_dir = os.path.join(output_root, "images")
        os.makedirs(images_dir, exist_ok=True)

        if color_image is None or depth_image is None:
            color_image, depth_image = self.get_data()

        rgb_path = os.path.join(images_dir, f"rgb_{frame_index:03d}.png")
        depth_path = os.path.join(images_dir, f"depth_{frame_index:03d}.npy")
        timestamps_path = os.path.join(images_dir, "timestamps.csv")

        ok = cv2.imwrite(rgb_path, color_image)
        if not ok:
            raise RuntimeError(f"Failed to save RGB image to {rgb_path}")
        np.save(depth_path, depth_image)

        ts = int(time.time_ns() if timestamp_ns is None else timestamp_ns)
        need_header = not os.path.exists(timestamps_path)
        with open(timestamps_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write("frame_idx,timestamp_ns\n")
            f.write(f"{frame_index:03d},{ts}\n")

        return {
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "timestamps_path": timestamps_path,
        }


def test() -> None:
    """测试函数"""
    camera = RealSenseCamera(width=1280, height=720)

    print("按 'q' 退出...")
    while True:
        camera.display(wait_key=1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.disconnect()
    camera.close()


if __name__ == "__main__":
    test()
