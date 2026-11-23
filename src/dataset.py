"""Load the prepared geometry and aerodynamic arrays for evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


class Standardizer:
    """Apply the saved per-column mean and scale to model inputs and outputs."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    @classmethod
    def from_dict(cls, data: dict) -> Standardizer:
        return cls(
            np.asarray(data["mean"], dtype=np.float32), np.asarray(data["std"], dtype=np.float32)
        )

    def encode(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std

    def decode(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std + self.mean


class AFBenchArrays:
    """Memory-map sample and geometry arrays linked by a sample-to-shape index."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.shape_vectors = np.load(self.root / "shape_vectors.npy", mmap_mode="r")
        self.coords = np.load(self.root / "coords.npy", mmap_mode="r")
        self.sample_shape_index = np.load(self.root / "sample_shape_index.npy", mmap_mode="r")
        self.aero_vectors = np.load(self.root / "aero_vectors.npy", mmap_mode="r")
        norm = json.loads((self.root / "normalization.json").read_text(encoding="utf-8"))
        self.shape_norm = Standardizer.from_dict(norm["shape"])
        self.aero_norm = Standardizer.from_dict(norm["aero"])
        self.shape_columns = norm["shape_columns"]
        self.aero_columns = norm["aero_columns"]

    def split_indices(self, split: str) -> np.ndarray:
        cache = self.root / f"{split}_indices.npy"
        if cache.exists():
            return np.load(cache, mmap_mode="r")

        samples_csv = self.root / "samples.csv"
        selected: list[int] = []
        with samples_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                split_col = header.index("split")
            except ValueError as exc:
                raise ValueError(f"{samples_csv} does not contain a split column") from exc
            for idx, row in enumerate(reader):
                if row[split_col] == split:
                    selected.append(idx)
        arr = np.asarray(selected, dtype=np.int64)
        tmp = cache.with_suffix(cache.suffix + ".tmp")
        with tmp.open("wb") as f:
            np.save(f, arr)
        tmp.replace(cache)
        return np.load(cache, mmap_mode="r")
