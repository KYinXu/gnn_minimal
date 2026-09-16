#!/usr/bin/env python3
"""
Run inference using a trained GNN model.

Usage:
  python inference.py --model_path results/gnn_models/run_.../gat_QSAR --input path/to/dataset.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.artifacts import (
    GNN_PLATT_SCALING,
    model_checkpoint_path,
    platt_scaling_path,
    resolve_model_dir,
    tabular_scaler_path,
)
from core.checkpoint_meta import (
    load_peptide_gnn_meta,
    resolve_node_layout_for_checkpoint,
)
from core.data_utils import PeptideGraphDataset
from core.extra_feature_scaler import load_extra_feature_scaler
from core.models import PeptideGNN, esm2_raw_dim_from_state_dict, esm2_hidden_dim_from_state_dict
from core.platt import load_platt_json, platt_prob_amp
from process_data import (
    ensure_processed_data,
    input_has_labels,
    resolve_input_and_generated,
)


def get_predictions_and_probs(model, loader, device, platt_payload=None):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            if platt_payload is not None:
                margins = (out[:, 1] - out[:, 0]).cpu().numpy()
                probs = platt_prob_amp(margins, platt_payload["coef"], platt_payload["intercept"])
            else:
                probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)


def load_processed_features(
    geometric_csv: Path,
    qsar_csv: Path,
    feature_cols: list[str],
) -> pd.DataFrame:
    df = pd.read_csv(geometric_csv)
    missing = [column for column in feature_cols if column not in df.columns]
    if missing:
        if not qsar_csv.is_file():
            raise FileNotFoundError(
                f"Tabular features {missing} are missing and QSAR file was not found: {qsar_csv}"
            )
        qsar_df = pd.read_csv(qsar_csv)
        qsar_cols = [column for column in missing if column in qsar_df.columns]
        if qsar_cols:
            df = df.merge(
                qsar_df[["peptide_id"] + qsar_cols],
                on="peptide_id",
                how="left",
            )
    remaining = [column for column in feature_cols if column not in df.columns]
    if remaining:
        raise ValueError(f"Processed data is missing tabular features: {remaining}")
    return df


def resolve_tabular_contract(model_dir: Path, meta: dict) -> tuple[Path | None, list[str]]:
    scaler_path = tabular_scaler_path(model_dir)
    feature_cols = list(meta.get("tabular_feature_cols") or [])
    expected_dim = int(meta.get("geo_feature_dim", len(feature_cols)))

    if not scaler_path.is_file():
        if expected_dim:
            raise FileNotFoundError(
                f"Checkpoint expects {expected_dim} tabular features but scaler is missing: {scaler_path}"
            )
        return None, []

    scaler = load_extra_feature_scaler(str(scaler_path))
    if feature_cols and feature_cols != scaler.feature_cols:
        raise ValueError("Checkpoint metadata and tabular scaler feature columns disagree")
    feature_cols = list(scaler.feature_cols)
    if expected_dim != len(feature_cols):
        raise ValueError(
            f"Checkpoint expects {expected_dim} tabular features; scaler has {len(feature_cols)}"
        )
    return scaler_path, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Test a saved GNN model on a new dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model directory (containing gnn_model.pt) or path to gnn_model.pt",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="CSV with id,sequence (label optional), or an existing generated/ folder",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_predictions", type=str, default="predictions.csv", help="Path to save predictions")
    parser.add_argument("--force-process", action="store_true", help="Regenerate processed features before inference")
    parser.add_argument("--device", type=str, default=None, help="Device for on-demand processing (default: auto)")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model_dir = resolve_model_dir(args.model_path)
        ckpt_path = model_checkpoint_path(model_dir)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    csv_path, gen_dir = resolve_input_and_generated(args.input)
    has_labels = bool(csv_path) and input_has_labels(csv_path)

    sd0 = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    esm2_raw = esm2_raw_dim_from_state_dict(sd0)
    esm2_h = esm2_hidden_dim_from_state_dict(sd0)

    try:
        gen_dir = ensure_processed_data(
            csv_path,
            gen_dir,
            need_esm2=esm2_raw > 0,
            require_labels=False,
            device=process_device,
            force=args.force_process,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    geo_csv = gen_dir / "geometric_features.csv"
    qsar_csv = gen_dir / "qsar12_descriptors.csv"
    pdb_dir = gen_dir / "structures"
    esm2_residue_dir = gen_dir / "esm2_per_residue"

    if not geo_csv.exists():
        print(f"Error: Processed data not found after processing: {geo_csv}")
        sys.exit(1)

    if esm2_raw > 0 and not esm2_residue_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint expects per-residue ESM2 but directory is missing: {esm2_residue_dir}"
        )

    meta = load_peptide_gnn_meta(model_dir)
    if meta is not None:
        architecture = meta.get("architecture", "gat")
        hidden_channels = meta.get("hidden_channels", 64)
        num_layers = meta.get("num_layers", 3)
        pooling = meta.get("pooling", "mean_max")
        num_classes = int(meta.get("num_classes", 2))
    else:
        print("Warning: model_summary.json not found, assuming defaults.")
        meta = {}
        architecture = "gat"
        hidden_channels = 64
        num_layers = 3
        pooling = "mean_max"
        num_classes = 2
    if num_classes != 2:
        raise ValueError("This AMP inference CLI supports binary checkpoints only")

    ng_infer, in_base_ckpt, layout_notes = resolve_node_layout_for_checkpoint(
        model_dir, sd0, architecture, user_node_groups=None
    )
    for note in layout_notes:
        print(note)

    scaler_path, feature_cols = resolve_tabular_contract(model_dir, meta)
    processed_df = load_processed_features(geo_csv, qsar_csv, feature_cols)
    dataset = PeptideGraphDataset(
        csv_path=str(geo_csv),
        pdb_dir=str(pdb_dir),
        use_geometric_features=bool(feature_cols),
        geometric_feature_cols=feature_cols,
        tabular_scaler_path=str(scaler_path) if scaler_path else None,
        esm2_residue_dir=str(esm2_residue_dir) if esm2_raw > 0 else None,
        node_feature_groups=ng_infer,
        dataframe=processed_df,
    )

    geo_dim = 0
    if len(dataset) > 0 and hasattr(dataset[0], "geo_features"):
        geo_dim = int(dataset[0].geo_features.shape[1])
    expected_geo_dim = int(meta.get("geo_feature_dim", geo_dim))
    if geo_dim != expected_geo_dim:
        raise ValueError(
            f"Tabular width mismatch: checkpoint expects {expected_geo_dim}, dataset produced {geo_dim}"
        )
    if len(dataset) > 0 and int(dataset[0].x.shape[1]) != in_base_ckpt:
        raise ValueError(
            f"Node width mismatch: checkpoint expects {in_base_ckpt}, "
            f"dataset produced {int(dataset[0].x.shape[1])}"
        )

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = PeptideGNN(
        architecture=architecture,
        in_channels=in_base_ckpt,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_classes=num_classes,
        pooling=pooling,
        geo_feature_dim=geo_dim,
        esm2_raw_dim=esm2_raw,
        esm2_hidden_dim=esm2_h,
    )

    model.load_state_dict(sd0)
    model = model.to(device)

    platt_payload = load_platt_json(platt_scaling_path(model_dir))
    if platt_payload:
        print(f"Loaded Platt scaling from {GNN_PLATT_SCALING}")

    probs = get_predictions_and_probs(model, test_loader, device, platt_payload)
    preds = (probs >= 0.5).astype(int)

    df = dataset.df
    ids = df["peptide_id"].astype(str).tolist()

    out = {"id": ids, "predicted_class": preds, "prob_AMP": probs}
    if has_labels and "label" in df.columns:
        out["true_label"] = df["label"].values

    pd.DataFrame(out).to_csv(args.save_predictions, index=False)
    print(f"\nSaved predictions to {args.save_predictions}")


if __name__ == "__main__":
    main()
