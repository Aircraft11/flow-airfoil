"""Rebuild the input arrays for the 20-coefficient CST representation from the AFBench archives."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .geometry import read_dat_bytes, resample_surfaces

SPLIT_SAMPLES = {"train": 2_760_152, "val": 343_647, "test": 344_634}
CST_X = np.linspace(0.0, 1.0, 101, dtype=np.float32)


def read_splits(path: Path) -> list[dict[str, str]]:
    """Validate archive-member identity, split membership by ID, and label linkage."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    members = {row["shape_member"] for row in rows}
    memberships = {(row["airfoil_id"], row["split"]) for row in rows}
    labels = [row["airfoil_id"] for row in rows if row["used_for_labels"] == "1"]
    if (
        len(rows) != len(members)
        or len(rows) != 125_208
        or len(memberships) != len({row["airfoil_id"] for row in rows})
        or len(labels) != len(set(labels))
        or len(labels) != 102_280
        or set(row["split"] for row in rows) != set(SPLIT_SAMPLES)
        or set(row["used_for_labels"] for row in rows) != {"0", "1"}
    ):
        raise ValueError(f"Invalid AFBench shape membership in {path}")
    return rows


@lru_cache(maxsize=1)
def cst_projection() -> np.ndarray:
    """Precompute the least-squares projection shared by both surfaces."""
    x = CST_X.astype(np.float64)
    order = 9
    class_function = np.sqrt(x) * (1.0 - x)
    bernstein = np.column_stack(
        [
            math.comb(order, i) * np.power(x, i) * np.power(1.0 - x, order - i)
            for i in range(order + 1)
        ]
    )
    return np.linalg.pinv(class_function[:, None] * bernstein)


def fit_cst20(coords: np.ndarray) -> np.ndarray:
    """Fit ten weights per surface on the study's uniform chordwise grid."""
    leading = int(np.argmin(coords[:, 0]))
    upper = coords[: leading + 1][::-1]
    lower = coords[leading:]
    projection = cst_projection()
    return np.concatenate(
        [projection @ np.interp(CST_X, surface[:, 0], surface[:, 1]) for surface in (upper, lower)]
    ).astype(np.float32)


def process_shape(item: tuple[int, str, bytes]) -> tuple[int, np.ndarray, np.ndarray]:
    index, name, payload = item
    try:
        _, raw = read_dat_bytes(payload, airfoil_id=Path(name).stem)
        # Reproduce the contour used in the study before fitting on CST_X.
        _, upper, lower = resample_surfaces(raw, n_points=101)
        coords = np.vstack([upper[::-1], lower[1:]]).astype(np.float32)
        return index, coords, fit_cst20(coords)
    except (ValueError, FloatingPointError) as exc:
        raise ValueError(f"Could not read AFBench shape {name}") from exc


def shape_inputs(archive: Path, rows: list[dict[str, str]]):
    missing = {row["shape_member"]: index for index, row in enumerate(rows)}
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.isfile() and member.name in missing:
                yield missing.pop(member.name), member.name, stream.extractfile(member).read()
    if missing:
        raise ValueError(
            f"{len(missing)} required shapes are missing, including {next(iter(missing))}"
        )


def write_shapes(archive: Path, rows: list[dict[str, str]], output: Path, workers: int) -> None:
    # CSV row indices determine array order even when the tar member order differs.
    coords = np.empty((len(rows), 201, 2), dtype=np.float32)
    shapes = np.empty((len(rows), 20), dtype=np.float32)
    inputs = shape_inputs(archive, rows)
    if workers == 1:
        for index, xy, weights in map(process_shape, inputs):
            coords[index], shapes[index] = xy, weights
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while batch := list(islice(inputs, 1024)):
                for index, xy, weights in pool.map(process_shape, batch, chunksize=32):
                    coords[index], shapes[index] = xy, weights
    np.save(output / "coords.npy", coords)
    np.save(output / "shape_vectors.npy", shapes)


def polar_rows(payload: bytes):
    started = False
    for line in payload.decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("------"):
            started = True
            continue
        fields = line.split()
        if not started or len(fields) < 5:
            continue
        try:
            alpha, cl, cd, cm = (float(fields[i]) for i in (0, 1, 2, 4))
        except ValueError:
            continue
        if cd > 0 and all(map(math.isfinite, (alpha, cl, cd, cm))):
            yield alpha, cl, cd, cm


def write_labels(archive: Path, rows: list[dict[str, str]], output: Path) -> None:
    # AFBench contains repeated ID stems. Use the geometry linked to the labels in
    # the original processing and keep the original archive and polar row order.
    source = {
        row["airfoil_id"]: (index, row["split"])
        for index, row in enumerate(rows)
        if row["used_for_labels"] == "1"
    }
    count = sum(SPLIT_SAMPLES.values())
    aero = np.empty((count, 6), dtype=np.float32)
    indices = np.empty(count, dtype=np.int64)
    split_indices = {split: np.empty(n, dtype=np.int64) for split, n in SPLIT_SAMPLES.items()}
    offsets = dict.fromkeys(SPLIT_SAMPLES, 0)
    sample = 0
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            parts = member.name.replace("\\", "/").split("/")
            if not member.isfile() or len(parts) < 4 or parts[-2] not in source:
                continue
            try:
                mach = float(parts[-1])
            except ValueError:
                continue
            shape_index, split = source[parts[-2]]
            for alpha, cl, cd, cm in polar_rows(stream.extractfile(member).read()):
                offset = offsets[split]
                if offset == SPLIT_SAMPLES[split]:
                    raise ValueError(f"Too many aerodynamic records for the {split} split")
                # Match the six-column state used to train the published checkpoints.
                ld = cl / max(cd, 1e-9)
                aero[sample] = (
                    mach,
                    alpha,
                    cl,
                    np.log(cd),
                    cm,
                    np.log(max(abs(ld), 1e-6)) * np.sign(ld),
                )
                indices[sample] = shape_index
                split_indices[split][offset] = sample
                offsets[split] += 1
                sample += 1
    if offsets != SPLIT_SAMPLES:
        raise ValueError(f"AFBench split sizes differ from the study's data: {offsets}")
    np.save(output / "aero_vectors.npy", aero)
    np.save(output / "sample_shape_index.npy", indices)
    for split, values in split_indices.items():
        np.save(output / f"{split}_indices.npy", values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-tar", required=True, type=Path)
    parser.add_argument("--aero-tar", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--normalization", type=Path, default=Path("data/normalization.json"))
    parser.add_argument(
        "--workers", type=int, default=1, help="Processes for geometry preparation."
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    for option, path in (("--splits", args.splits), ("--normalization", args.normalization)):
        if not path.is_file():
            parser.error(f"{path} is missing. Supply an authorized copy with {option}.")
    rows = read_splits(args.splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Publish the dataset directory only after both array groups are complete.
    with TemporaryDirectory(prefix=".cst20-", dir=args.out.parent) as temporary:
        output = Path(temporary) / "arrays"
        output.mkdir()
        shutil.copyfile(args.normalization, output / "normalization.json")
        write_shapes(args.shape_tar, rows, output, args.workers)
        write_labels(args.aero_tar, rows, output)
        output.rename(args.out)
    print(f"Prepared {len(rows):,} geometries and {sum(SPLIT_SAMPLES.values()):,} label rows.")


if __name__ == "__main__":
    main()
