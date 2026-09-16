"""Canonical on-disk names for a trained gnn_minimal model directory."""

from __future__ import annotations

from pathlib import Path

GNN_MODEL_PT = "gnn_model.pt"
MODEL_SUMMARY_JSON = "model_summary.json"
GNN_PLATT_SCALING = "gnn_platt_scaling"
TABULAR_SCALER_JOBLIB = "tabular_scaler.joblib"


def resolve_model_dir(path: str | Path) -> Path:
    """Accept a model directory or a path to ``gnn_model.pt``."""
    path = Path(path).resolve()
    if path.is_dir():
        return path
    if path.is_file() and path.name == GNN_MODEL_PT:
        return path.parent
    if path.is_file() and path.suffix == ".pt":
        return path.parent
    raise FileNotFoundError(
        f"Expected a model directory containing {GNN_MODEL_PT}, or a path to that file; got: {path}"
    )


def model_checkpoint_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / GNN_MODEL_PT


def model_summary_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / MODEL_SUMMARY_JSON


def platt_scaling_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / GNN_PLATT_SCALING


def tabular_scaler_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / TABULAR_SCALER_JOBLIB
