"""Read and resample airfoil contours and decode shape parameters."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import comb

EPS = 1e-8


@dataclass(frozen=True)
class AirfoilGeometry:
    airfoil_id: str
    coords: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    x_grid: np.ndarray
    cst_upper: np.ndarray
    cst_lower: np.ndarray
    features: np.ndarray
    feature_names: list[str]


def read_dat_bytes(payload: bytes, airfoil_id: str | None = None) -> tuple[str, np.ndarray]:
    text = payload.decode("utf-8", errors="ignore")
    rows: list[list[float]] = []
    name = airfoil_id or "airfoil"
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                rows.append([float(parts[0]), float(parts[1])])
                continue
            except ValueError:
                pass
        if not rows and stripped:
            name = stripped.split()[0]
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 12 or arr.shape[1] != 2:
        raise ValueError(f"{name}: not enough coordinate rows")
    return name, normalize_chord(arr)


def read_dat_file(path: str | Path) -> tuple[str, np.ndarray]:
    p = Path(path)
    return read_dat_bytes(p.read_bytes(), airfoil_id=p.stem)


def normalize_chord(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    coords = coords[np.isfinite(coords).all(axis=1)]
    # Remove exact consecutive duplicates; they destabilize PCHIP near LE/TE.
    keep = np.ones(len(coords), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(coords, axis=0), axis=1) > 1e-12
    coords = coords[keep]
    x = coords[:, 0]
    y = coords[:, 1]
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    chord = max(x_max - x_min, EPS)
    out = np.column_stack(((x - x_min) / chord, y / chord))
    # Center trailing edge to y=0 while preserving thickness.
    te_mask = out[:, 0] > 0.995
    if np.any(te_mask):
        out[:, 1] -= float(np.mean(out[te_mask, 1]))
    return out


def split_surfaces(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = normalize_chord(coords)
    le_idx = int(np.argmin(coords[:, 0]))
    upper = coords[: le_idx + 1]
    lower = coords[le_idx:]
    if len(upper) < 4 or len(lower) < 4:
        # Fallback for files that list upper/lower in separate x-sorted blocks.
        mid = len(coords) // 2
        upper, lower = coords[:mid], coords[mid - 1 :]
    upper = _unique_sorted_by_x(upper)
    lower = _unique_sorted_by_x(lower)
    if np.nanmean(upper[:, 1]) < np.nanmean(lower[:, 1]):
        upper, lower = lower, upper
    return upper, lower


def _unique_sorted_by_x(points: np.ndarray) -> np.ndarray:
    order = np.argsort(points[:, 0], kind="mergesort")
    pts = points[order]
    xs = pts[:, 0]
    ys = pts[:, 1]
    ux: list[float] = []
    uy: list[float] = []
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and abs(xs[j] - xs[i]) < 1e-10:
            j += 1
        ux.append(float(np.mean(xs[i:j])))
        uy.append(float(np.mean(ys[i:j])))
        i = j
    arr = np.column_stack([ux, uy])
    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
    return arr


def resample_surfaces(
    coords: np.ndarray, n_points: int = 101
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper, lower = split_surfaces(coords)
    x_grid = cosine_x_grid(n_points)
    up = _interp_surface(upper, x_grid)
    lo = _interp_surface(lower, x_grid)
    return x_grid, up, lo


def cosine_x_grid(n_points: int = 101) -> np.ndarray:
    """Cluster chordwise points near the leading and trailing edges."""
    beta = np.linspace(0.0, math.pi, n_points)
    return 0.5 * (1.0 - np.cos(beta))


def _interp_surface(surface: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    x = np.clip(surface[:, 0], 0.0, 1.0)
    y = surface[:, 1]
    if len(np.unique(np.round(x, 12))) < 4:
        raise ValueError("surface has fewer than four unique x coordinates")
    interp = PchipInterpolator(x, y, extrapolate=True)
    yy = interp(x_grid)
    yy[0] = float(np.mean(y[x < 0.01])) if np.any(x < 0.01) else yy[0]
    yy[-1] = float(np.mean(y[x > 0.99])) if np.any(x > 0.99) else yy[-1]
    return np.column_stack([x_grid, yy])


def bernstein_basis(x: np.ndarray, order: int, n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
    """Evaluate class-weighted Bernstein polynomials for one surface."""
    x = np.clip(np.asarray(x, dtype=np.float64), EPS, 1.0 - EPS)
    cls = np.power(x, n1) * np.power(1.0 - x, n2)
    cols = []
    for i in range(order + 1):
        cols.append(comb(order, i) * np.power(x, i) * np.power(1.0 - x, order - i) * cls)
    return np.column_stack(cols)


def fit_cst(surface: np.ndarray, order: int = 7) -> np.ndarray:
    x = np.clip(surface[:, 0], EPS, 1.0 - EPS)
    y = surface[:, 1]
    basis = bernstein_basis(x, order)
    coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return coeffs.astype(np.float32)


def cst_to_surface(coeffs: np.ndarray, x_grid: np.ndarray | None = None) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if x_grid is None:
        x_grid = cosine_x_grid(101)
    basis = bernstein_basis(x_grid, len(coeffs) - 1)
    y = basis @ coeffs
    return np.column_stack([x_grid, y])


def shape_from_cst(cst_upper: np.ndarray, cst_lower: np.ndarray, n_points: int = 101) -> np.ndarray:
    x_grid = cosine_x_grid(n_points)
    upper = cst_to_surface(cst_upper, x_grid)
    lower = cst_to_surface(cst_lower, x_grid)
    # AFBench .dat convention: upper TE -> LE, then lower LE -> TE.
    return np.vstack([upper[::-1], lower[1:]])


def cst_shape_vector_to_coords(shape: np.ndarray, n_points: int = 101) -> np.ndarray:
    """Decode a concatenation of upper- and lower-surface CST coefficients."""
    shape = np.asarray(shape, dtype=np.float64)
    half = shape.shape[-1] // 2
    return shape_from_cst(shape[:half], shape[half:], n_points=n_points)


def _parsec_surface_coefficients(
    leading_edge_radius: float,
    crest_x: float,
    crest_y: float,
    crest_curvature: float,
    te_y: float,
    te_slope: float,
    upper: bool,
) -> np.ndarray:
    """Recover the six classical PARSEC polynomial coefficients for one surface.

    The surface is represented as ``y=sum_i a_i x^(i-1/2)`` for i=1..6.
    The leading-edge coefficient is fixed by the leading-edge radius; the
    remaining coefficients are solved from the crest ordinate, zero crest
    slope, crest curvature, trailing-edge ordinate and trailing-edge slope.
    """

    r_le = max(float(leading_edge_radius), 1e-10)
    x_c = float(np.clip(crest_x, 1e-4, 0.999))
    a1 = math.sqrt(2.0 * r_le)
    if not upper:
        a1 = -a1

    powers = np.asarray([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)
    # Unknowns are a2..a6. The a1*x^0.5 contribution is moved to the RHS.
    mat = np.vstack(
        [
            np.power(x_c, powers),
            powers * np.power(x_c, powers - 1.0),
            powers * (powers - 1.0) * np.power(x_c, powers - 2.0),
            np.ones_like(powers),
            powers,
        ]
    )
    rhs = np.asarray(
        [
            float(crest_y) - a1 * math.sqrt(x_c),
            -0.5 * a1 / math.sqrt(x_c),
            float(crest_curvature) + 0.25 * a1 / np.power(x_c, 1.5),
            float(te_y) - a1,
            float(te_slope) - 0.5 * a1,
        ],
        dtype=np.float64,
    )
    try:
        tail = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        tail, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
    return np.concatenate([[a1], tail])


def parsec_to_surfaces(parsec: np.ndarray, n_points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Convert AFBench 11-D PARSEC annotation to upper/lower coordinate surfaces.

    Parameter order follows AFBench: ``Rle, Xup, Yup, Zxxup, Xlo, Ylo,
    Zxxlo, Yte, deltaYte, alpha_te, beta_te``. Angles are treated as radians.
    Coordinate reconstruction follows the AirfoilRAG/AFBench half-power
    PARSEC convention, ``y=sum_i a_i x^(i-1/2)`` for i=1..6, with the
    coefficients fixed by leading-edge radius, crest point, crest curvature,
    trailing-edge ordinate and trailing-edge angle constraints.
    In the AFBench files inspected for this project, ``alpha_te`` behaves as
    the upper-surface trailing-edge angle and ``beta_te`` behaves as the
    lower-surface trailing-edge angle. They are therefore not decoded as the
    classical PARSEC mean trailing-edge angle and included wedge angle.
    The public AFBench labels typically have zero trailing-edge thickness for
    the CST-generated subset, but non-zero values are supported.
    """

    p = np.asarray(parsec, dtype=np.float64).reshape(-1)
    if p.shape[0] != 11:
        raise ValueError(f"PARSEC vector must have length 11, got {p.shape[0]}")
    rle, xup, yup, zxxup, xlo, ylo, zxxlo, yte, dyte, alpha_te, beta_te = p
    x_grid = cosine_x_grid(n_points)
    upper_slope = math.tan(float(alpha_te))
    lower_slope = math.tan(float(beta_te))
    upper_te_y = float(yte) + 0.5 * float(dyte)
    lower_te_y = float(yte) - 0.5 * float(dyte)
    coeff_up = _parsec_surface_coefficients(
        rle, xup, yup, zxxup, upper_te_y, upper_slope, upper=True
    )
    coeff_lo = _parsec_surface_coefficients(
        rle, xlo, ylo, zxxlo, lower_te_y, lower_slope, upper=False
    )
    exponents = np.asarray([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)
    xx = np.clip(x_grid, 0.0, 1.0)[:, None]
    basis = np.power(xx, exponents[None, :])
    upper = np.column_stack([x_grid, basis @ coeff_up])
    lower = np.column_stack([x_grid, basis @ coeff_lo])
    return upper.astype(np.float32), lower.astype(np.float32)


def parsec_shape_vector_to_coords(parsec: np.ndarray, n_points: int = 101) -> np.ndarray:
    upper, lower = parsec_to_surfaces(parsec, n_points=n_points)
    return np.vstack([upper[::-1], lower[1:]]).astype(np.float32)


def shape_vector_to_coords(
    shape: np.ndarray, shape_columns: list[str] | None = None, n_points: int = 101
) -> np.ndarray:
    shape = np.asarray(shape, dtype=np.float64)
    if shape_columns and all(str(name).startswith("parsec_") for name in shape_columns):
        return parsec_shape_vector_to_coords(shape, n_points=n_points)
    if shape_columns and str(shape_columns[0]).startswith("cst_upper_weight_"):
        if len(shape_columns) != len(shape):
            raise ValueError("CST columns do not match the geometry vector")
        thickness = str(shape_columns[-1]) == "cst_trailing_edge_thickness"
        count = (len(shape) - int(thickness)) // 2
        expected = [f"cst_upper_weight_{i:02d}" for i in range(count)]
        expected += [f"cst_lower_weight_{i:02d}" for i in range(count)]
        if thickness:
            expected.append("cst_trailing_edge_thickness")
        if list(shape_columns) != expected or count == 0:
            raise ValueError("Invalid named CST column layout")
        coords = shape_from_cst(shape[:count], shape[count : 2 * count], n_points=n_points)
        coords[[0, n_points - 1, -1], 1] = [0.0, 0.0, 0.0]
        if thickness:
            coords[:n_points, 1] += 0.5 * coords[:n_points, 0] * shape[-1]
            coords[n_points:, 1] -= 0.5 * coords[n_points:, 0] * shape[-1]
        return coords
    if shape.shape[-1] == 11:
        return parsec_shape_vector_to_coords(shape, n_points=n_points)
    return cst_shape_vector_to_coords(shape, n_points=n_points)


def extract_geometry(
    coords: np.ndarray, airfoil_id: str, n_points: int = 101, cst_order: int = 7
) -> AirfoilGeometry:
    x_grid, upper, lower = resample_surfaces(coords, n_points=n_points)
    cst_upper = fit_cst(upper, order=cst_order)
    cst_lower = fit_cst(lower, order=cst_order)
    thickness = upper[:, 1] - lower[:, 1]
    camber = 0.5 * (upper[:, 1] + lower[:, 1])
    dy_up = np.gradient(upper[:, 1], x_grid)
    dy_lo = np.gradient(lower[:, 1], x_grid)
    curv_up = np.gradient(dy_up, x_grid) / np.power(1.0 + dy_up**2, 1.5)
    curv_lo = np.gradient(dy_lo, x_grid) / np.power(1.0 + dy_lo**2, 1.5)
    i_t = int(np.argmax(thickness))
    i_c = int(np.argmax(np.abs(camber)))
    feature_names = [
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
    upper_arc = float(np.sum(np.linalg.norm(np.diff(upper, axis=0), axis=1)))
    lower_arc = float(np.sum(np.linalg.norm(np.diff(lower, axis=0), axis=1)))
    area = float(np.trapezoid(thickness, x_grid))
    features = np.asarray(
        [
            float(thickness[i_t]),
            float(x_grid[i_t]),
            float(camber[i_c]),
            float(x_grid[i_c]),
            float(np.max(np.abs(camber))),
            float(thickness[-1]),
            upper_arc,
            lower_arc,
            float(np.trapezoid(np.abs(curv_up), x_grid)),
            float(np.trapezoid(np.abs(curv_lo), x_grid)),
            float(1.0 / max(abs(curv_up[1]), EPS)),
            area,
        ],
        dtype=np.float32,
    )
    coords_rs = np.vstack([upper[::-1], lower[1:]]).astype(np.float32)
    return AirfoilGeometry(
        airfoil_id=airfoil_id,
        coords=coords_rs,
        upper=upper.astype(np.float32),
        lower=lower.astype(np.float32),
        x_grid=x_grid.astype(np.float32),
        cst_upper=cst_upper,
        cst_lower=cst_lower,
        features=features,
        feature_names=feature_names,
    )


def coords_to_dat_text(coords: np.ndarray, name: str = "generated_airfoil") -> str:
    buf = io.StringIO()
    buf.write(f"{name}\n")
    for x, y in np.asarray(coords):
        buf.write(f"{x:.10f} {y:.10f}\n")
    return buf.getvalue()
