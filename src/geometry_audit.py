"""Nested coordinate audit used to evaluate the main model."""

import numpy as np

from .constraints import airfoil_quality_metrics, geometry_penalty
from .geometry import resample_surfaces, shape_vector_to_coords

GEOMETRY_AUDIT_PROFILES = ("legacy", "cst_tight")


def audit_candidates(shapes, columns, *, profile="cst_tight", geometry_points=181):
    """Decode candidate geometries and report nested compliance and quality."""
    compliant = np.zeros(len(shapes), dtype=bool)
    quality = np.zeros_like(compliant)
    rejects = {}
    decode_failures = 0
    for i, shape in enumerate(shapes):
        if not np.isfinite(shape).all():
            decode_failures += 1
            continue
        try:
            points = shape_vector_to_coords(shape, columns, n_points=geometry_points)
        except (ValueError, FloatingPointError):
            decode_failures += 1
            continue
        if not np.isfinite(points).all():
            decode_failures += 1
            continue
        _, a = geometric_compliance_eval(points, profile=profile)
        _, b = coordinate_geometry_eval(points, profile=profile)
        compliant[i] = a["valid"]
        # The published coordinate-quality rate requires passing both levels.
        quality[i] = a["valid"] and b["valid"]
        for flag in set(a["flags"] + b["flags"]):
            rejects[flag] = rejects.get(flag, 0) + 1
    return compliant, quality, {"decode_failures": decode_failures, "flags": rejects}


def _turn_count(values: np.ndarray, x_grid: np.ndarray, threshold: float) -> int:
    """Count slope sign changes after suppressing weak slopes and edge effects."""
    deriv = np.gradient(values, x_grid)
    mask = (x_grid > 0.04) & (x_grid < 0.96)
    signs = np.sign(deriv[mask])
    signs[np.abs(deriv[mask]) < threshold] = 0.0
    compressed: list[float] = []
    for sign in signs:
        if sign == 0:
            continue
        if not compressed or compressed[-1] != float(sign):
            compressed.append(float(sign))
    return max(0, len(compressed) - 1)


def publication_shape_metrics(coords: np.ndarray) -> dict[str, float]:
    """Measure trailing-edge taper, post-peak thickening, camber, and oscillation."""
    x_grid, upper, lower = resample_surfaces(coords, n_points=181)
    yu = upper[:, 1].astype(float)
    yl = lower[:, 1].astype(float)
    thickness = yu - yl
    max_t = max(float(np.max(thickness)), 1e-8)
    i_t = int(np.argmax(thickness))
    post = thickness[i_t:]
    aft = thickness[x_grid >= 0.70]
    camber = 0.5 * (yu + yl)
    dyu = np.gradient(yu, x_grid)
    dyl = np.gradient(yl, x_grid)
    mid = (x_grid > 0.04) & (x_grid < 0.96)
    return {
        "te_abs_y": float(max(abs(yu[-1]), abs(yl[-1]))),
        "tail_thickness_ratio": float(abs(thickness[-1]) / max_t),
        "post_max_thickness_increase": float(np.sum(np.maximum(np.diff(post), 0.0)))
        if len(post) > 1
        else 0.0,
        "aft70_thickness_increase": float(np.sum(np.maximum(np.diff(aft), 0.0)))
        if len(aft) > 1
        else 0.0,
        "max_camber": float(np.max(np.abs(camber))),
        "thickness_turns": float(_turn_count(thickness, x_grid, threshold=0.015)),
        "upper_turns": float(_turn_count(yu, x_grid, threshold=0.020)),
        "lower_turns": float(_turn_count(yl, x_grid, threshold=0.020)),
        "max_mid_slope": float(max(np.max(np.abs(dyu[mid])), np.max(np.abs(dyl[mid]))))
        if np.any(mid)
        else 0.0,
    }


def coordinate_geometry_eval(
    coords: np.ndarray, *, profile: str = "legacy"
) -> tuple[dict[str, float], dict[str, object]]:
    """Apply the stricter coordinate-quality thresholds to one contour."""
    if profile not in GEOMETRY_AUDIT_PROFILES:
        raise ValueError(f"Unknown geometry audit profile: {profile}")
    tight = profile == "cst_tight"
    metrics = airfoil_quality_metrics(coords)
    metrics.update(publication_shape_metrics(coords))
    geom = geometry_penalty(
        metrics,
        feature_bounds=None,
        min_thickness=0.0005,
        max_te_thickness=0.0075 if tight else 0.008,
        max_curvature_l1=14.0 if tight else 15.0,
    )
    flags = list(geom["flags"])
    score = float(geom["score"])
    valid = bool(geom["valid"])
    checks = [
        (
            0.040 <= float(metrics["max_thickness"]) <= (0.210 if tight else 0.220),
            "max_thickness_range",
        ),
        (0.10 <= float(metrics["max_thickness_x"]) <= 0.65, "max_thickness_location"),
        (float(metrics["le_radius_proxy"]) >= 0.015, "leading_edge_radius_floor"),
        (abs(float(metrics["max_camber"])) <= (0.09 if tight else 0.10), "camber_range"),
        (float(metrics["tail_thickness_ratio"]) <= 0.50, "tail_taper"),
        (float(metrics["post_max_thickness_increase"]) <= 0.025, "post_peak_monotonicity"),
        (float(metrics["aft70_thickness_increase"]) <= 0.015, "aft_monotonicity"),
        (int(metrics["thickness_turns"]) <= 4, "thickness_turns"),
        (int(metrics["upper_turns"]) <= 5 and int(metrics["lower_turns"]) <= 5, "surface_turns"),
        (float(metrics["max_mid_slope"]) <= (2.20 if tight else 2.25), "mid_chord_slope"),
    ]
    for ok, name in checks:
        if not ok:
            valid = False
            flags.append(name)
            score += 100.0
    return metrics, {"score": score, "valid": valid, "flags": flags}


def geometric_compliance_eval(
    coords: np.ndarray, *, profile: str = "legacy"
) -> tuple[dict[str, float], dict[str, object]]:
    """Apply the first-level geometric feasibility thresholds to one contour."""
    if profile not in GEOMETRY_AUDIT_PROFILES:
        raise ValueError(f"Unknown geometry audit profile: {profile}")
    tight = profile == "cst_tight"
    metrics = airfoil_quality_metrics(coords)
    metrics.update(publication_shape_metrics(coords))
    geom = geometry_penalty(
        metrics,
        feature_bounds=None,
        min_thickness=0.0,
        max_te_thickness=0.014 if tight else 0.015,
        max_curvature_l1=24.0 if tight else 25.0,
    )
    flags = list(geom["flags"])
    score = float(geom["score"])
    valid = bool(geom["valid"])
    checks = [
        (float(metrics["le_radius_proxy"]) >= 0.005, "leading_edge_radius_floor"),
        (float(metrics["tail_thickness_ratio"]) <= (0.70 if tight else 0.75), "tail_taper"),
        (int(metrics["thickness_turns"]) <= 6, "thickness_turns"),
        (int(metrics["upper_turns"]) <= 8 and int(metrics["lower_turns"]) <= 8, "surface_turns"),
    ]
    for ok, name in checks:
        if not ok:
            valid = False
            flags.append(name)
            score += 100.0
    return metrics, {"score": score, "valid": valid, "flags": flags}
