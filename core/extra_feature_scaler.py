"""
Per-source RobustScaler for GNN tabular extras (geo_features).

Fits sklearn RobustScaler independently on geometric, QSAR, and ESM2 column blocks
using training rows only, then rescales each block so median |value| matches across
blocks (reduces dominance of large-raw-scale columns like SASA vs embedding dims).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

GNN_GEO_COLS: List[str] = [
    "radius_gyration",
    "end_to_end_distance",
    "max_pairwise_distance",
    "centroid_distance_mean",
    "centroid_distance_std",
    "fraction_helix",
    "fraction_sheet",
    "fraction_coil",
    "total_sasa",
    "hydrophobic_sasa",
    "fraction_hydrophobic_sasa",
    "length",
    "net_charge",
    "mean_hydrophobicity",
    "hydrophobic_moment",
    "curvature_mean",
    "curvature_std",
    "curvature_max",
    "torsion_mean",
    "torsion_std",
]

GNN_QSAR_COLS: List[str] = [
    "netCharge",
    "FC",
    "LW",
    "DP",
    "NK",
    "AE",
    "pcMK",
    "_SolventAccessibilityD1025",
    "tau2_GRAR740104",
    "tau4_GRAR740104",
    "QSO50_GRAR740104",
    "QSO29_GRAR740104",
]


def partition_feature_blocks(
    feature_cols: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Split ordered feature_cols into geo, qsar, esm2 blocks (training column order)."""
    geo_set = set(GNN_GEO_COLS)
    qsar_set = set(GNN_QSAR_COLS)
    geo = [c for c in feature_cols if c in geo_set]
    qsar = [c for c in feature_cols if c in qsar_set]
    esm = [c for c in feature_cols if c not in geo_set and c not in qsar_set]
    return geo, qsar, esm


@dataclass
class _Block:
    col_names: List[str]
    scaler: RobustScaler
    block_weight: float


class ExtraFeatureRobustScaler:
    """Apply block-wise RobustScaler + optional per-block magnitude balancing."""

    def __init__(
        self,
        feature_cols: List[str],
        blocks: List[_Block],
    ):
        self.feature_cols = list(feature_cols)
        self.blocks = blocks
        self._col_index = {c: i for i, c in enumerate(self.feature_cols)}

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        feature_cols: Sequence[str],
        *,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        balance_blocks: bool = True,
        block_weight_clip: Tuple[float, float] = (1e-4, 1e4),
    ) -> "ExtraFeatureRobustScaler":
        feature_cols = list(feature_cols)
        if not feature_cols:
            raise ValueError("feature_cols is empty")
        geo, qsar, esm = partition_feature_blocks(feature_cols)
        block_specs: List[Tuple[str, List[str]]] = []
        if geo:
            block_specs.append(("geo", geo))
        if qsar:
            block_specs.append(("qsar", qsar))
        if esm:
            block_specs.append(("esm2", esm))
        flat = [c for _, cols in block_specs for c in cols]
        if set(flat) != set(feature_cols) or len(flat) != len(feature_cols):
            raise ValueError(
                "feature_cols must be a union of geo, qsar, and non-geo/non-qsar (ESM) "
                f"columns with no extras; got {feature_cols!r}"
            )

        blocks: List[_Block] = []
        lo, hi = block_weight_clip
        for _, cols in block_specs:
            X = train_df[cols].to_numpy(dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0)
            scaler = RobustScaler(
                with_centering=True,
                with_scaling=True,
                quantile_range=quantile_range,
                unit_variance=False,
            )
            Xt = scaler.fit_transform(X)
            if balance_blocks:
                med = float(np.median(np.abs(Xt)))
                if med < 1e-12:
                    w = 1.0
                else:
                    w = 1.0 / med
                w = float(np.clip(w, lo, hi))
            else:
                w = 1.0
            blocks.append(_Block(list(cols), scaler, w))
        return cls(feature_cols, blocks)

    def transform_vector(self, vec: np.ndarray) -> np.ndarray:
        """vec: 1d, same order as feature_cols."""
        x = np.asarray(vec, dtype=np.float64).ravel()
        if x.shape[0] != len(self.feature_cols):
            raise ValueError(
                f"Expected {len(self.feature_cols)} values, got {x.shape[0]}"
            )
        x = np.nan_to_num(x, nan=0.0)
        out = np.zeros_like(x)
        for b in self.blocks:
            idx = [self._col_index[c] for c in b.col_names]
            sub = x[idx]
            sub_t = b.scaler.transform(sub.reshape(1, -1)).ravel() * b.block_weight
            for j, ii in enumerate(idx):
                out[ii] = sub_t[j]
        return out

    def transform_row(self, row: pd.Series) -> np.ndarray:
        raw = row[self.feature_cols].to_numpy(dtype=np.float64)
        return self.transform_vector(raw)


def save_extra_feature_scaler(scaler: ExtraFeatureRobustScaler, path: str) -> None:
    import joblib

    joblib.dump(scaler, path)


def load_extra_feature_scaler(path: str) -> ExtraFeatureRobustScaler:
    import joblib

    obj = joblib.load(path)
    if not isinstance(obj, ExtraFeatureRobustScaler):
        raise TypeError(f"Not an ExtraFeatureRobustScaler: {type(obj)}")
    return obj
