#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Capture RGB-D frames for SVD experiments and save them into data/svd."""

from __future__ import annotations

import os
import sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from handeye.camera.realsense import RealSenseCamera
from handeye.config import get_svd_data_path


def _next_index(images_dir: str) -> int:
    if not os.path.isdir(images_dir):
        return 1

    indices = []
    for name in os.listdir(images_dir):
        if not (name.startswith("rgb_") and name.endswith(".png")):
            continue
        try:
            idx = int(name.split("_")[1].split(".")[0])
        except (ValueError, IndexError):
            continue
        indices.append(idx)

    if not indices:
        return 1
    return max(indices) + 1


def main() -> None:
    paths = get_svd_data_path()
    os.makedirs(paths["images"], exist_ok=True)

    camera = RealSenseCamera()
    camera.connect()
    intr_path = camera.write_intrinsics_snapshot(paths["root"])
    print(f"intrinsics snapshot saved: {intr_path}")

    next_idx = _next_index(paths["images"])
    print("\nSVD capture started")
    print("Space: save current frame")
    print("Q or Esc: quit")

    try:
        while True:
            color, depth = camera.get_data()
            preview = color.copy()
            cv2.putText(preview, f"next idx: {next_idx:03d}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(preview, "Space=save  Q/Esc=quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("SVD RGB-D Capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                out = camera.save_rgbd_frame(
                    frame_index=next_idx,
                    output_root=paths["root"],
                    color_image=color,
                    depth_image=depth,
                )
                print(f"saved frame {next_idx:03d}: {out['rgb_path']}, {out['depth_path']}")
                next_idx += 1
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
