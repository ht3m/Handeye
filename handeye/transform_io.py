"""Read and write hand-eye transform matrices."""

from __future__ import annotations

import ast

import numpy as np


def format_transform_matrix(transform: np.ndarray, precision: int = 6) -> str:
    """Format a 4x4 transform as a Python-style nested list."""
    mat = np.asarray(transform, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {mat.shape}")

    rows = []
    for row_index, row in enumerate(mat):
        if row_index == 3 and np.allclose(row, [0.0, 0.0, 0.0, 1.0]):
            values = "0.0, 0.0, 0.0, 1.0"
        else:
            values = ", ".join(f"{float(value):.{precision}f}" for value in row)
        rows.append(f"    [{values}],")
    return "[\n" + "\n".join(rows) + "\n]\n"


def save_transform_matrix(path: str, transform: np.ndarray, precision: int = 6) -> None:
    """Save a 4x4 transform in a readable nested-list format."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_transform_matrix(transform, precision=precision))


def load_transform_matrix(path: str) -> np.ndarray:
    """Load a 4x4 transform from either nested-list or whitespace format."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if text.startswith("["):
        mat = np.asarray(ast.literal_eval(text), dtype=np.float64)
    else:
        mat = np.asarray(np.loadtxt(path), dtype=np.float64)

    if mat.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform in {path}, got {mat.shape}")
    return mat
