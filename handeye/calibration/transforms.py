#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common rigid transform helpers used across calibration modules."""

from typing import Sequence, Union
import numpy as np


def vec_to_skew(v: np.ndarray) -> np.ndarray:
    """Convert a 3D vector to a 3x3 skew-symmetric matrix."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def skew_to_vec(skew: np.ndarray) -> np.ndarray:
    """Convert a 3x3 skew-symmetric matrix to its 3D vector form."""
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)


def rotvec_to_matrix(rotvec: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert a rotation vector to a 3x3 rotation matrix."""
    v = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)

    axis = v / theta
    K = vec_to_skew(axis)
    R = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return np.asarray(R, dtype=np.float64)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a rotation vector."""
    Rn = np.asarray(R, dtype=np.float64).reshape(3, 3)
    theta = float(np.arccos(np.clip((np.trace(Rn) - 1.0) / 2.0, -1.0, 1.0)))
    if abs(theta) < 1e-8:
        return np.zeros(3, dtype=np.float64)

    log_R = (theta / (2.0 * np.sin(theta))) * (Rn - Rn.T)
    return skew_to_vec(log_R)


def pose_to_mat(pose: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert [x, y, z, rx, ry, rz] pose to a 4x4 homogeneous transform."""
    pose_arr = np.asarray(pose, dtype=np.float64)
    if pose_arr.shape == (4, 4):
        return pose_arr.copy()

    if pose_arr.shape != (6,):
        raise ValueError(f"pose must be shape (6,) or (4,4), got {pose_arr.shape}")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotvec_to_matrix(pose_arr[3:6])
    T[:3, 3] = pose_arr[:3]
    return T


def mat_to_pose(T: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogeneous transform to [x, y, z, rx, ry, rz]."""
    Tn = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return np.concatenate([Tn[:3, 3], matrix_to_rotvec(Tn[:3, :3])])


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transform."""
    Tn = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = Tn[:3, :3]
    t = Tn[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
