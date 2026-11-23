"""Compute coordinate-derived metrics and apply basic geometric limits."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .geometry import resample_surfaces

FEATURE_NAMES = [
    "max_thickness",
    "max_thickness_x",
    "max_camber",
    "max_camber_x",
    "abs_max_camber",
    "te_thickness",
    "upper_arc",
    "lower_arc",
    "upper_curvature_l1",
    "lower_curvature_l1",
    "le_radius_proxy",
    "area",
]


def airfoil_quality_metrics(coords: np.ndarray, n_points: int = 101) -> dict[str, float]:
    x_grid, upper, lower = resample_surfaces(coords, n_points=n_points)
    thickness = upper[:, 1] - lower[:, 1]
    camber = 0.5 * (upper[:, 1] + lower[:, 1])
    dy_up = np.gradient(upper[:, 1], x_grid)
    dy_lo = np.gradient(lower[:, 1], x_grid)
    curv_up = np.gradient(dy_up, x_grid) / np.power(1.0 + dy_up**2, 1.5)
    curv_lo = np.gradient(dy_lo, x_grid) / np.power(1.0 + dy_lo**2, 1.5)
    coords_rs = np.vstack([upper[::-1], lower[1:]])
    d1 = np.diff(coords_rs, axis=0)
    d2 = np.diff(d1, axis=0)
    i_t = int(np.argmax(thickness))
    i_c = int(np.argmax(np.abs(camber)))
    upper_arc = float(np.sum(np.linalg.norm(np.diff(upper, axis=0), axis=1)))
    lower_arc = float(np.sum(np.linalg.norm(np.diff(lower, axis=0), axis=1)))
    area = float(np.trapezoid(thickness, x_grid))
    # Interior thickness checks exclude the nearly coincident endpoints.
    interior = (x_grid > 0.01) & (x_grid < 0.99)
    interior_thickness = thickness[interior] if np.any(interior) else thickness
    interior_x = x_grid[interior] if np.any(interior) else x_grid
    min_t_idx = int(np.argmin(interior_thickness))
    return {
        "max_thickness": float(thickness[i_t]),
        "max_thickness_x": float(x_grid[i_t]),
        "max_camber": float(camber[i_c]),
        "max_camber_x": float(x_grid[i_c]),
        "abs_max_camber": float(np.max(np.abs(camber))),
        "te_thickness": float(thickness[-1]),
        "upper_arc": upper_arc,
        "lower_arc": lower_arc,
        "upper_curvature_l1": float(np.trapezoid(np.abs(curv_up), x_grid)),
        "lower_curvature_l1": float(np.trapezoid(np.abs(curv_lo), x_grid)),
        "le_radius_proxy": float(1.0 / max(abs(curv_up[1]), 1e-8)),
        "area": area,
        "min_thickness": float(interior_thickness[min_t_idx]),
        "min_thickness_x": float(interior_x[min_t_idx]),
        "smoothness": float(np.mean(np.linalg.norm(d2, axis=1))) if len(d2) else 0.0,
        "upper_arc_lower_arc_ratio": float(upper_arc / max(lower_arc, 1e-8)),
    }


def load_feature_bounds(
    root: str | Path,
    lower_q: float = 0.005,
    upper_q: float = 0.995,
    feature_names: list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    root = Path(root)
    path = root / "airfoils.csv"
    names = feature_names or FEATURE_NAMES
    if not path.exists():
        coords_path = root / "coords.npy"
        if not coords_path.exists():
            return {}
        coords = np.load(coords_path, mmap_mode="r")
        values = {name: [] for name in names}
        max_airfoils = min(len(coords), 20000)
        if max_airfoils <= 0:
            return {}
        indices = np.linspace(0, len(coords) - 1, max_airfoils).round().astype(np.int64)
        for idx in indices:
            try:
                metrics = airfoil_quality_metrics(np.asarray(coords[int(idx)], dtype=np.float64))
            except Exception:
                continue
            for name in names:
                val = metrics.get(name)
                if val is not None and np.isfinite(val):
                    values[name].append(float(val))
        bounds: dict[str, tuple[float, float]] = {}
        for name, vals in values.items():
            if len(vals) < 8:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            lo = float(np.quantile(arr, lower_q))
            hi = float(np.quantile(arr, upper_q))
            if np.isfinite([lo, hi]).all() and hi > lo:
                bounds[name] = (lo, hi)
        return bounds
    values = {name: [] for name in names}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for name in names:
                raw = row.get(name)
                if raw in (None, ""):
                    continue
                try:
                    val = float(raw)
                except ValueError:
                    continue
                if np.isfinite(val):
                    values[name].append(val)
    bounds: dict[str, tuple[float, float]] = {}
    for name, vals in values.items():
        if len(vals) < 8:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        lo = float(np.quantile(arr, lower_q))
        hi = float(np.quantile(arr, upper_q))
        if np.isfinite([lo, hi]).all() and hi > lo:
            bounds[name] = (lo, hi)
    return bounds


def geometry_penalty(
    metrics: Mapping[str, float],
    feature_bounds: Mapping[str, tuple[float, float]] | None = None,
    min_thickness: float = 1e-4,
    max_te_thickness: float | None = None,
    max_curvature_l1: float | None = None,
) -> dict[str, object]:
    penalty = 0.0
    flags: list[str] = []
    min_t = float(metrics.get("min_thickness", -np.inf))
    if min_t <= min_thickness:
        scale = max(abs(min_thickness), 1e-4)
        penalty += float(((min_thickness - min_t) / scale) ** 2)
        flags.append("surface_crossing_or_nonpositive_thickness")
    te_t = abs(float(metrics.get("te_thickness", 0.0)))
    if max_te_thickness is not None and te_t > max_te_thickness:
        scale = max(max_te_thickness, 1e-4)
        penalty += float(((te_t - max_te_thickness) / scale) ** 2)
        flags.append("large_trailing_edge_thickness")
    if max_curvature_l1 is not None:
        curv = max(
            float(metrics.get("upper_curvature_l1", 0.0)),
            float(metrics.get("lower_curvature_l1", 0.0)),
        )
        if curv > max_curvature_l1:
            penalty += float(((curv - max_curvature_l1) / max(max_curvature_l1, 1e-4)) ** 2)
            flags.append("large_curvature")
    if feature_bounds:
        for name, (lo, hi) in feature_bounds.items():
            if name not in metrics:
                continue
            val = float(metrics[name])
            scale = max(hi - lo, 1e-8)
            if val < lo:
                penalty += float(((lo - val) / scale) ** 2)
                flags.append(f"{name}_below_training_range")
            elif val > hi:
                penalty += float(((val - hi) / scale) ** 2)
                flags.append(f"{name}_above_training_range")
    return {
        "score": float(penalty),
        "valid": len(flags) == 0,
        "flags": flags,
    }
