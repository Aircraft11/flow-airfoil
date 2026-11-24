"""Generate CST airfoil DAT files from a checkpoint and physical targets."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .dataset import Standardizer
from .geometry import coords_to_dat_text, cst_shape_vector_to_coords
from .geometry_audit import audit_candidates
from .main_model import DEFAULT_TEMPERATURE, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True, help="Angle of attack in degrees")
    parser.add_argument("--cl", type=float, required=True)
    parser.add_argument("--cd", type=float, required=True)
    parser.add_argument("--cm", type=float, required=True)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, help="Optional sampling seed")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    values = (args.mach, args.alpha, args.cl, args.cd, args.cm, args.temperature)
    if (
        not all(map(math.isfinite, values))
        or not 0 <= args.mach < 1
        or args.cd <= 0
        or args.temperature < 0
    ):
        parser.error("Use finite targets, 0 <= Mach < 1, positive drag and nonnegative temperature")
    if args.candidates < 1 or args.cpu_threads < 1 or args.out_dir.exists():
        parser.error("Use positive counts and a new output directory")

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    model, saved = load_model(args.checkpoint, device)
    aero_norm = Standardizer.from_dict(saved["normalization"]["aero"])
    shape_norm = Standardizer.from_dict(saved["normalization"]["shape"])
    ld = args.cl / max(args.cd, 1e-9)
    target = np.array(
        [
            [
                args.mach,
                args.alpha,
                args.cl,
                np.log(args.cd),
                args.cm,
                np.log(max(abs(ld), 1e-6)) * np.sign(ld),
            ]
        ],
        dtype=np.float32,
    )
    aero = torch.as_tensor(aero_norm.encode(target), device=device).expand(args.candidates, -1)
    rng = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    with torch.inference_mode():
        noise = torch.randn((args.candidates, model.flow.latent_dim), device=device, generator=rng)
        shapes = shape_norm.decode(model.inverse_flow(noise * args.temperature, aero).cpu().numpy())
    if not np.isfinite(shapes).all():
        raise RuntimeError("Nonfinite CST coefficients from checkpoint")

    points = 181
    compliance, quality, _ = audit_candidates(
        shapes, saved["shape_columns"], geometry_points=points
    )
    args.out_dir.mkdir(parents=True)
    records = []
    for i, shape in enumerate(shapes):
        name = f"candidate_{i + 1:03d}"
        coords = cst_shape_vector_to_coords(shape, n_points=points)
        (args.out_dir / f"{name}.dat").write_text(
            coords_to_dat_text(coords, name), encoding="ascii"
        )
        records.append(
            {
                "file": f"{name}.dat",
                "cst_coefficients": shape.tolist(),
                "geometric_compliance": bool(compliance[i]),
                "coordinate_quality": bool(quality[i]),
            }
        )
    report = {
        "mach": args.mach,
        "alpha": args.alpha,
        "cl": args.cl,
        "cd": args.cd,
        "cm": args.cm,
        "temperature": args.temperature,
        "candidates": records,
    }
    (args.out_dir / "candidates.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
