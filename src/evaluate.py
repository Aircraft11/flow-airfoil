"""Evaluate forward prediction, cycle consistency and generated geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import AFBenchArrays
from .geometry_audit import audit_candidates
from .main_model import DEFAULT_TEMPERATURE, load_model

TARGET_DRAW = 20260729
DIAGNOSTIC_DRAW = 20260508
CANDIDATES_PER_TARGET = 4


def physical(value, arrays):
    """Undo the aerodynamic standardization and convert log drag to CD."""
    p = np.asarray(value, dtype=np.float64) * arrays.aero_norm.std[2:5] + arrays.aero_norm.mean[2:5]
    with np.errstate(over="ignore", invalid="ignore"):
        p[:, 1] = np.exp(p[:, 1])
    return p


def reg_metrics(pred, truth):
    finite = np.isfinite(pred).all(axis=1) & np.isfinite(truth).all(axis=1)
    result = {"n": len(truth), "finite": int(finite.sum()), "coefficients": {}}
    for col, name in enumerate(("cl", "cd", "cm")):
        p, t = pred[finite, col], truth[finite, col]
        e = np.abs(p - t)
        variance_sum = float(np.sum((t - t.mean()) ** 2)) if len(t) else 0.0
        result["coefficients"][name] = (
            dict(
                r2=1 - float((e**2).sum()) / variance_sum if variance_sum > 0 else None,
                mae=float(e.mean()),
                p95=float(np.quantile(e, 0.95)),
                p99=float(np.quantile(e, 0.99)),
            )
            if len(e)
            else dict.fromkeys(("r2", "mae", "p95", "p99"))
        )
    return result


def select_targets(split, count, draw):
    if len(split) < count:
        raise ValueError(f"Requested {count} from a split of {len(split)}")
    return np.sort(np.random.default_rng(draw).choice(split, size=count, replace=False))


def run_batches(
    model, arrays, indices, batch_size, device, sample=False, observed=False, draw=TARGET_DRAW
):
    rng = torch.Generator(device=device).manual_seed(draw)
    aero_raw = np.asarray(arrays.aero_vectors[indices], dtype=np.float32)
    aero_norm = arrays.aero_norm.encode(aero_raw)
    samples, recoveries, source = [], [], []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            sl = slice(start, start + batch_size)
            aero = torch.as_tensor(aero_norm[sl], device=device)
            if sample:
                # Generate in one direction and recover the requested coefficients with the same model.
                latent = torch.randn(
                    (len(aero), int(model.flow.latent_dim)), device=device, generator=rng
                )
                latent = latent * DEFAULT_TEMPERATURE
                shapes = model.inverse_flow(latent, aero)
                recovered = model.predict_aero(shapes, aero)
                samples.append(shapes.cpu().numpy())
                recoveries.append(recovered.cpu().numpy())
            if observed:
                shape_ids = arrays.sample_shape_index[indices[sl]]
                shape = arrays.shape_norm.encode(
                    np.asarray(arrays.shape_vectors[shape_ids], dtype=np.float32)
                )
                recovered = model.predict_aero(torch.as_tensor(shape, device=device), aero)
                source.append(recovered.cpu().numpy())
    output = {"aero_raw": aero_raw, "target_norm": aero_norm[:, 2:5], "sample_indices": indices}
    if sample:
        output["shapes_raw"] = arrays.shape_norm.decode(np.concatenate(samples))
        output["recovered_norm"] = np.concatenate(recoveries)
    if observed:
        output["source_pred_norm"] = np.concatenate(source)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/afbench_cst20"))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.batch_size < 1 or args.cpu_threads < 1 or args.out_dir.exists():
        parser.error("Use positive batch and thread counts and a new output directory")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    arrays = AFBenchArrays(args.data)
    model, saved = load_model(args.checkpoint, device)
    for key in ("shape_columns", "aero_columns"):
        if getattr(arrays, key) != saved[key]:
            raise ValueError(f"Data and checkpoint differ in {key}")
    for kind, standardizer in (("shape", arrays.shape_norm), ("aero", arrays.aero_norm)):
        for key in ("mean", "std"):
            if not np.array_equal(
                getattr(standardizer, key),
                np.asarray(saved["normalization"][kind][key], dtype=np.float32),
            ):
                raise ValueError(f"Data and checkpoint differ in {kind} {key}")

    test = arrays.split_indices("test")
    if len(np.unique(test)) != len(test):
        raise ValueError("Test split contains duplicate record indices")

    # Geometry audits four candidates per selected target. Both regression
    # diagnostics use the complete test split in its stored row order.
    targets = select_targets(test, 512, TARGET_DRAW)
    candidates = run_batches(
        model,
        arrays,
        np.repeat(targets, CANDIDATES_PER_TARGET),
        args.batch_size,
        device,
        sample=True,
    )
    compliance, quality, detail = audit_candidates(candidates["shapes_raw"], arrays.shape_columns)
    candidates.update(
        candidate_indices=np.tile(np.arange(CANDIDATES_PER_TARGET), len(targets)),
        geometric_compliance=compliance,
        coordinate_quality=quality,
    )
    observed = run_batches(model, arrays, test, args.batch_size, device, observed=True)
    cycle = run_batches(
        model, arrays, test, args.batch_size, device, sample=True, draw=DIAGNOSTIC_DRAW
    )
    target_phys = physical(observed["target_norm"], arrays)
    observed_phys = physical(observed["source_pred_norm"], arrays)
    cycle_phys = physical(cycle["recovered_norm"], arrays)
    report = {
        "test_indices_sha256": hashlib.sha256(np.asarray(test, dtype="<i8").tobytes()).hexdigest(),
        "observed_geometry": reg_metrics(observed_phys, target_phys),
        "diagnostic_cycle": reg_metrics(cycle_phys, target_phys),
        "geometry": {
            "n": len(compliance),
            "geometric_compliance": int(compliance.sum()),
            "coordinate_quality": int(quality.sum()),
            **detail,
        },
    }
    if any(
        report[key]["finite"] != report[key]["n"]
        for key in ("observed_geometry", "diagnostic_cycle")
    ):
        raise RuntimeError("Nonfinite aerodynamic predictions")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out_dir / "candidates.npz", **candidates)
    np.savez_compressed(
        args.out_dir / "test_predictions.npz",
        sample_indices=test,
        target_phys=target_phys,
        observed_phys=observed_phys,
        cycle_phys=cycle_phys,
    )
    (args.out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
