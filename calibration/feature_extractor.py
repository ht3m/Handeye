#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征提取模块 - ArUco 标记检测 & 棋盘格角点检测
用于手眼标定的特征提取
"""

import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Tuple, cast


class ArucoDetector:
    """ArUco 标记检测器，用于估计标记相对于相机的位姿"""

    def __init__(
        self,
        dictionary_name: str = 'DICT_ARUCO_ORIGINAL',
        marker_size: float = 0.10,
        marker_id: int = 996
    ) -> None:
        """
        初始化 ArUco 检测器

        Args:
            dictionary_name: OpenCV ArUco 字典名称
            marker_size: 标记边长 (米)
            marker_id: 目标标记 ID
        """
        self.marker_size = marker_size
        self.marker_id = marker_id

        # 获取 ArUco 字典
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dictionary_name)
        )
        self.aruco_dict = aruco_dict

        # 创建检测参数
        self.detector_params = cv2.aruco.DetectorParameters()

        # 创建检测器 (OpenCV 4.7+)
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.detector_params
        )

    def detect_marker(
        self,
        image: np.ndarray,
        intrinsics: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        检测 ArUco 标记并估计位姿

        Args:
            image: BGR 彩色图像
            intrinsics: 相机内参矩阵 (3, 3)
            dist_coeffs: 相机畸变系数 (可选)

        Returns:
            dict: {
                'success': bool,
                'tag_pose': 4x4 齐次变换矩阵 (标记在相机坐标系下),
                'tag_corners': 标记角点像素坐标 (4, 2),
                'tag_id': 检测到的标记 ID,
                'image': 原始图像
            }
        """
        if dist_coeffs is None:
            dist_coeffs_np = np.zeros((5, 1), dtype=np.float64)
        else:
            dist_coeffs_np = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

        # 检测标记
        corners, ids, _ = self.detector.detectMarkers(image)

        result: Dict[str, Any] = {
            'success': False,
            'tag_pose': None,
            'tag_corners': None,
            'tag_id': None,
            'image': image
        }

        if ids is None or len(ids) == 0:
            return result

        # 查找目标 ID
        target_idx = None
        for i, tag_id in enumerate(ids):
            if int(tag_id[0]) == self.marker_id:
                target_idx = i
                break

        if target_idx is None:
            return result

        # 使用 solvePnP 估计位姿
        # ArUco 标记的世界坐标 (以标记中心为原点, Z轴垂直于标记平面)
        half = self.marker_size / 2.0
        obj_points = np.array([
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0]
        ], dtype=np.float64)

        tag_corners = corners[target_idx].reshape(-1, 2).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            obj_points,
            tag_corners,
            intrinsics,
            dist_coeffs_np,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok:
            return result

        # 转换为 4x4 齐次变换矩阵
        R, _ = cv2.Rodrigues(rvec)
        tag_pose = np.eye(4, dtype=np.float64)
        tag_pose[:3, :3] = R
        tag_pose[:3, 3] = tvec.reshape(3)

        result['success'] = True
        result['tag_pose'] = tag_pose
        result['tag_corners'] = tag_corners
        result['tag_id'] = int(ids[target_idx][0])
        return result


class CheckerboardExtractor:
    """棋盘格角点提取器"""

    def __init__(self, checkerboard_size: Tuple[int, int] = (5, 5), square_size: float = 0.024) -> None:
        """
        初始化角点提取器

        Args:
            checkerboard_size: 棋盘格内角点数量 (cols, rows)
            square_size: 方格大小 (米)
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size

        # 亚像素角点优化参数
        self.refine_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30, 0.001
        )

    def detect_corners(
        self,
        gray_image: np.ndarray,
        refine: bool = True
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        检测棋盘格角点

        Args:
            gray_image: 灰度图
            refine: 是否进行亚像素级优化

        Returns:
            success: 检测是否成功
            corners: 角点坐标 (N, 1, 2)
            corners_refined: 亚像素级角点 (refine=True)
        """
        success, corners = cv2.findChessboardCorners(
            gray_image,
            self.checkerboard_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not success:
            return False, None, None

        if refine:
            corners_refined = cv2.cornerSubPix(
                gray_image,
                corners,
                (5, 5),
                (-1, -1),
                self.refine_criteria
            )
            return True, corners, corners_refined

        return True, corners, corners

    def get_center_pixel(self, corners: np.ndarray) -> np.ndarray:
        """
        获取棋盘格中心像素坐标

        Args:
            corners: 角点坐标

        Returns:
            center_px: 中心像素坐标 [u, v]
        """
        center_idx = (self.checkerboard_size[0] * self.checkerboard_size[1]) // 2
        center_corner = corners[center_idx, 0]
        center_px = np.round(center_corner).astype(int)
        return np.asarray(center_px, dtype=np.int32)

    def get_corners_3d(
        self,
        corners: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: np.ndarray
    ) -> np.ndarray:
        """
        将角点像素坐标转换为3D相机坐标

        Args:
            corners: 角点像素坐标 (N, 1, 2)
            depth_image: 深度图 (H, W), 单位: 米
            intrinsics: 相机内参矩阵 (3, 3)

        Returns:
            corners_3d: 角点3D坐标 (N, 3)
        """
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        corners_3d = []

        for i in range(len(corners)):
            u = int(np.round(corners[i, 0, 0]))
            v = int(np.round(corners[i, 0, 1]))

            if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                z = depth_image[v, u]
                if z > 0:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    corners_3d.append([x, y, z])
                else:
                    corners_3d.append([0, 0, 0])
            else:
                corners_3d.append([0, 0, 0])

        return np.array(corners_3d)

    def estimate_board_pose(self, corners_3d: np.ndarray) -> np.ndarray:
        """
        估计棋盘格板相对于相机的位姿

        Args:
            corners_3d: 棋盘格角点在相机坐标系下的3D坐标

        Returns:
            H_cb: 棋盘格相对于相机的齐次变换矩阵 (4, 4)
        """
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        centered = corners_3d - corners_3d.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]

        if normal[2] > 0:
            normal = -normal

        z_axis = normal / np.linalg.norm(normal)
        x_axis = np.cross([0, 0, 1], z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1, 0, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])
        t = corners_3d.mean(axis=0)

        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t

        return H

    def extract(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: np.ndarray
    ) -> Dict[str, Any]:
        """
        提取棋盘格特征 (完整流程)

        Args:
            color_image: 彩色图 (BGR)
            depth_image: 深度图 (米)
            intrinsics: 相机内参

        Returns:
            result: 包含以下键的字典:
                - success: 是否成功
                - corners_2d: 角点2D坐标
                - corners_3d: 角点3D坐标
                - center_px: 中心像素坐标
                - board_pose: 棋盘格位姿 (4x4)
                - visualized: 可视化图像
        """
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        success, corners, corners_refined = self.detect_corners(gray, refine=True)

        if not success:
            return {'success': False, 'message': '未检测到棋盘格'}
        assert corners_refined is not None

        center_px = self.get_center_pixel(corners_refined)

        u, v = center_px
        if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
            depth = depth_image[v, u]
        else:
            depth = 0

        if depth <= 0:
            return {'success': False, 'message': '深度值无效'}

        corners_3d = self.get_corners_3d(corners_refined, depth_image, intrinsics)
        board_pose = self.estimate_board_pose(corners_3d)

        vis = color_image.copy()
        cv2.drawChessboardCorners(vis, self.checkerboard_size, corners_refined, success)
        start_pt = tuple(corners_refined[0, 0].astype(int))
        end_pt = tuple(corners_refined[-1, 0].astype(int))
        cv2.circle(vis, start_pt, 9, (0, 255, 255), -1)
        cv2.putText(vis, 'START', (start_pt[0] + 8, start_pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(vis, end_pt, 9, (255, 0, 255), -1)
        cv2.putText(vis, 'END', (end_pt[0] + 8, end_pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.circle(vis, tuple(center_px), 10, (0, 255, 0), -1)

        return {
            'success': True,
            'corners_2d': corners_refined,
            'corners_3d': corners_3d,
            'center_px': center_px,
            'center_depth': depth,
            'board_pose': board_pose,
            'visualized': vis
        }


def visualize_checkerboard(
    color_image: np.ndarray,
    corners: np.ndarray,
    checkerboard_size: Tuple[int, int],
    center_idx: Optional[int] = None
) -> np.ndarray:
    """
    可视化棋盘格检测结果

    Args:
        color_image: 彩色图
        corners: 角点坐标
        checkerboard_size: 棋盘格大小
        center_idx: 中心点索引

    Returns:
        vis: 可视化图像
    """
    vis = color_image.copy()
    cv2.drawChessboardCorners(vis, checkerboard_size, corners, True)

    if corners is not None and len(corners) > 0:
        start_pt = tuple(corners[0, 0].astype(int))
        end_pt = tuple(corners[-1, 0].astype(int))
        cv2.circle(vis, start_pt, 9, (0, 255, 255), -1)
        cv2.putText(vis, 'START', (start_pt[0] + 8, start_pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(vis, end_pt, 9, (255, 0, 255), -1)
        cv2.putText(vis, 'END', (end_pt[0] + 8, end_pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    if center_idx is not None:
        center = corners[center_idx, 0]
        cv2.circle(vis, tuple(center.astype(int)), 15, (0, 255, 0), -1)

    return vis


def load_and_process(
    image_path: str,
    depth_path: str,
    intrinsics: np.ndarray,
    checkerboard_size: Tuple[int, int] = (5, 5)
) -> Dict[str, Any]:
    """
    加载并处理图像

    Args:
        image_path: 彩色图路径
        depth_path: 深度图路径 (.npy)
        intrinsics: 相机内参
        checkerboard_size: 棋盘格大小

    Returns:
        result: 处理结果字典
    """
    color_img = cv2.imread(image_path)
    if color_img is None:
        return {'success': False, 'message': f'无法读取图像: {image_path}'}
    depth_img = np.load(depth_path)

    extractor = CheckerboardExtractor(checkerboard_size)
    result = extractor.extract(color_img, depth_img, intrinsics)

    return result


if __name__ == "__main__":
    extractor = CheckerboardExtractor(checkerboard_size=(5, 5))
    print("棋盘格提取器已创建")
    print(f"  角点数量: {extractor.checkerboard_size}")
    print(f"  方格大小: {extractor.square_size}m")