#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SVD-based offline feature extraction for saved RGB-D frames."""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from handeye.config import get_svd_data_path


@dataclass
class SVDFrameFeatures:
    """Compact per-frame SVD statistics for RGB and depth channels."""

    frame_idx: int
    rgb_energy_ratio_1: float
    rgb_energy_ratio_2: float
    rgb_energy_ratio_3: float
    depth_energy_ratio_1: float
    depth_energy_ratio_2: float
    depth_energy_ratio_3: float


def _resize_for_svd(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image
    scale = float(max_side) / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _energy_ratios(matrix: np.ndarray, top_k: int = 3) -> List[float]:
    singular_values = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    energies = singular_values * singular_values
    total_energy = float(np.sum(energies))
    if total_energy <= 1e-12:
        return [0.0] * top_k

    ratios: List[float] = []
    for i in range(top_k):
        if i >= energies.shape[0]:
            ratios.append(0.0)
        else:
            ratios.append(float(energies[i] / total_energy))
    return ratios


def _build_frame_index(images_dir: str) -> List[int]:
    rgb_files = glob.glob(os.path.join(images_dir, "rgb_*.png"))
    indices: List[int] = []
    for path in rgb_files:
        name = os.path.basename(path)
        try:
            idx = int(name.split("_")[1].split(".")[0])
        except (ValueError, IndexError):
            continue
        indices.append(idx)
    return sorted(set(indices))


def _load_depth_for_svd(depth_path: str, max_side: int) -> Optional[np.ndarray]:
    if not os.path.exists(depth_path):
        return None

    depth = np.asarray(np.load(depth_path), dtype=np.float64)
    if depth.ndim != 2:
        return None

    valid = np.isfinite(depth) & (depth > 0.0)
    if int(np.count_nonzero(valid)) < 100:
        return None

    median_depth = float(np.median(depth[valid]))
    depth_filled = depth.copy()
    depth_filled[~valid] = median_depth
    depth_small = _resize_for_svd(depth_filled, max_side)
    depth_centered = depth_small - float(np.mean(depth_small))
    return depth_centered


def _load_rgb_for_svd(rgb_path: str, max_side: int) -> Optional[np.ndarray]:
    if not os.path.exists(rgb_path):
        return None

    gray = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.ndim != 2:
        return None

    gray_small = _resize_for_svd(gray, max_side)
    gray_f = gray_small.astype(np.float64)
    gray_centered = gray_f - float(np.mean(gray_f))
    return gray_centered


def analyze_saved_frames(
    images_dir: str,
    output_csv_path: str,
    max_side: int = 320,
) -> List[SVDFrameFeatures]:
    """Run SVD feature extraction for RGB and depth frames saved in one folder."""
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    results: List[SVDFrameFeatures] = []
    indices = _build_frame_index(images_dir)

    for idx in indices:
        rgb_path = os.path.join(images_dir, f"rgb_{idx:03d}.png")
        depth_path = os.path.join(images_dir, f"depth_{idx:03d}.npy")

        rgb_matrix = _load_rgb_for_svd(rgb_path, max_side=max_side)
        depth_matrix = _load_depth_for_svd(depth_path, max_side=max_side)
        if rgb_matrix is None or depth_matrix is None:
            continue

        rgb_ratios = _energy_ratios(rgb_matrix, top_k=3)
        depth_ratios = _energy_ratios(depth_matrix, top_k=3)

        results.append(
            SVDFrameFeatures(
                frame_idx=idx,
                rgb_energy_ratio_1=rgb_ratios[0],
                rgb_energy_ratio_2=rgb_ratios[1],
                rgb_energy_ratio_3=rgb_ratios[2],
                depth_energy_ratio_1=depth_ratios[0],
                depth_energy_ratio_2=depth_ratios[1],
                depth_energy_ratio_3=depth_ratios[2],
            )
        )

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx",
            "rgb_energy_ratio_1",
            "rgb_energy_ratio_2",
            "rgb_energy_ratio_3",
            "depth_energy_ratio_1",
            "depth_energy_ratio_2",
            "depth_energy_ratio_3",
        ])
        for row in results:
            writer.writerow([
                row.frame_idx,
                row.rgb_energy_ratio_1,
                row.rgb_energy_ratio_2,
                row.rgb_energy_ratio_3,
                row.depth_energy_ratio_1,
                row.depth_energy_ratio_2,
                row.depth_energy_ratio_3,
            ])

    return results


def analyze_default_svd_data() -> List[SVDFrameFeatures]:
    """Analyze frames in data/svd/images and write data/svd/svd_features.csv."""
    paths: Dict[str, str] = get_svd_data_path()
    return analyze_saved_frames(paths["images"], paths["features"])


if __name__ == "__main__":
    rows = analyze_default_svd_data()
    print(f"SVD feature extraction complete: {len(rows)} frames")
