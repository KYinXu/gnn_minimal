"""Platt scaling: logistic calibration on GNN logit margin for P(AMP)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import DataLoader


@torch.no_grad()
def collect_margins_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    margins: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        target = batch.y.view(-1)
        if target.numel() == 0:
            continue
        m = (out[:, 1] - out[:, 0]).cpu().numpy()
        y = target.cpu().numpy()
        margins.append(m)
        labels.append(y)
    if not margins:
        return np.array([]), np.array([])
    return np.concatenate(margins), np.concatenate(labels)


def fit_platt(margins: np.ndarray, labels: np.ndarray) -> dict | None:
    if margins.size == 0 or len(np.unique(labels)) < 2:
        return None
    X = margins.reshape(-1, 1).astype(np.float64)
    y = labels.astype(np.int64)
    clf = LogisticRegression(solver="lbfgs", C=1e12, max_iter=2000)
    clf.fit(X, y)
    coef = float(clf.coef_.ravel()[0])
    intercept = float(clf.intercept_.ravel()[0])
    return {
        "method": "platt",
        "score": "logit_margin",
        "coef": coef,
        "intercept": intercept,
        "n_calib": int(len(y)),
    }


def platt_prob_amp(margins: np.ndarray, coef: float, intercept: float) -> np.ndarray:
    z = coef * margins.astype(np.float64) + intercept
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def default_platt_path(checkpoint_or_dir: str | Path) -> Path:
    from .artifacts import platt_scaling_path, resolve_model_dir

    return platt_scaling_path(resolve_model_dir(checkpoint_or_dir))


def save_platt_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_platt_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("method") != "platt" or "coef" not in data or "intercept" not in data:
        return None
    return data


def softmax_prob_amp(logits: torch.Tensor) -> np.ndarray:
    return F.softmax(logits, dim=1)[:, 1].cpu().numpy()
