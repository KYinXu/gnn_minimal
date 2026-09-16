#!/usr/bin/env python3
"""
Train single GNN models (no CV) for test-time inference.

Usage:
  python train.py path/to/labeled_dataset.csv
  python train.py path/to/generated/   # if already processed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data_utils import (
    NodeFeatureGroups,
    load_esm2_per_residue_tensor,
    node_input_dim,
    resolve_peptide_pdb_path,
    wants_esm2_residue_nodes,
)
from core.models import PeptideGNN
from core.platt import (
    collect_margins_and_labels,
    fit_platt,
    save_platt_json,
)
from core.train import run_training
from core.extra_feature_scaler import ExtraFeatureRobustScaler, save_extra_feature_scaler
from core.checkpoint_meta import save_peptide_gnn_meta
from core.artifacts import (
    model_checkpoint_path,
    platt_scaling_path,
    tabular_scaler_path,
)
from process_data import ensure_processed_data, input_has_labels, resolve_input_and_generated


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_labels(raw_labels) -> np.ndarray:
    return np.array([1 if int(label) == 1 else 0 for label in raw_labels], dtype=np.int64)


def require_both_classes(labels: np.ndarray, *, context: str) -> None:
    present = sorted(int(x) for x in np.unique(labels))
    if len(present) < 2:
        raise ValueError(
            f"{context}: need both classes for binary training, found only {present}. "
            "Check that your CSV has a 'label' column with AMP=1 and non-AMP=0 (or -1)."
        )


def load_data_with_features(csv_path: str, qsar_csv: str, source_csv: Path | None = None):
    geo_df = pd.read_csv(csv_path)
    qsar_df = pd.read_csv(qsar_csv)

    qsar_cols = [
        "netCharge", "FC", "LW", "DP", "NK", "AE", "pcMK",
        "_SolventAccessibilityD1025", "tau2_GRAR740104", "tau4_GRAR740104",
        "QSO50_GRAR740104", "QSO29_GRAR740104",
    ]

    merged_df = geo_df.merge(qsar_df[["peptide_id"] + qsar_cols], on="peptide_id", how="left")

    if source_csv is not None:
        source = pd.read_csv(source_csv)
        if "label" not in source.columns:
            raise ValueError(
                f"Training CSV must include a 'label' column: {source_csv}"
            )
        label_map = source[["id", "label"]].copy()
        label_map["id"] = label_map["id"].astype(str)
        if "label" in merged_df.columns:
            merged_df = merged_df.drop(columns=["label"])
        merged_df = merged_df.merge(
            label_map.rename(columns={"id": "peptide_id"}),
            on="peptide_id",
            how="left",
        )

    if "label" not in merged_df.columns or merged_df["label"].isna().any():
        raise ValueError(
            "Training data is missing labels. "
            "Provide a labeled CSV (id,label,sequence)."
        )
    require_both_classes(
        binary_labels(merged_df["label"].values),
        context="Loaded training labels",
    )
    return merged_df, qsar_cols


def create_feature_cols(use_geo: bool, use_qsar: bool, qsar_cols):
    geo_cols = [
        "radius_gyration", "end_to_end_distance", "max_pairwise_distance",
        "centroid_distance_mean", "centroid_distance_std",
        "fraction_helix", "fraction_sheet", "fraction_coil",
        "total_sasa", "hydrophobic_sasa", "fraction_hydrophobic_sasa",
        "length", "net_charge", "mean_hydrophobicity", "hydrophobic_moment",
        "curvature_mean", "curvature_std", "curvature_max",
        "torsion_mean", "torsion_std",
    ]
    cols = []
    if use_geo:
        cols.extend(geo_cols)
    if use_qsar and qsar_cols:
        cols.extend(qsar_cols)
    return cols


def infer_esm2_raw_dim(df: pd.DataFrame, esm2_residue_dir: Path) -> int:
    if not esm2_residue_dir.is_dir():
        raise FileNotFoundError(
            f"Per-residue ESM2 is enabled but directory is missing: {esm2_residue_dir}"
        )
    for peptide_id in df["peptide_id"]:
        tensor = load_esm2_per_residue_tensor(esm2_residue_dir, peptide_id)
        if tensor.ndim != 2:
            raise ValueError(
                f"ESM2 tensor for {peptide_id!r} must be rank 2, got {tensor.shape}"
            )
        return int(tensor.shape[1])
    raise ValueError("Cannot infer ESM2 width from an empty dataset")


class CustomPeptideDataset:
    def __init__(
        self,
        df,
        pdb_dir,
        feature_cols,
        distance_threshold: float = 8.0,
        tabular_scaler: ExtraFeatureRobustScaler | None = None,
        esm2_residue_dir: str | None = None,
        node_feature_groups: NodeFeatureGroups | None = None,
    ):
        self.df = df
        self.pdb_dir = Path(pdb_dir)
        self.feature_cols = feature_cols
        self.distance_threshold = distance_threshold
        self.tabular_scaler = tabular_scaler
        self.esm2_residue_dir = Path(esm2_residue_dir).resolve() if esm2_residue_dir else None
        self.node_feature_groups = node_feature_groups

        from core.data_utils import pdb_to_graph, parse_pdb, compute_node_features, compute_edges
        self.pdb_to_graph = pdb_to_graph
        self.parse_pdb = parse_pdb
        self.compute_node_features = compute_node_features
        self.compute_edges = compute_edges

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from torch_geometric.data import Data
        from core.data_utils import parse_pdb, compute_node_features, compute_edges, load_esm2_per_residue_tensor

        row = self.df.iloc[idx]
        pdb_file = row.get("pdb_file", None)
        pdb_path = resolve_peptide_pdb_path(self.pdb_dir, pdb_file, row["peptide_id"])

        if pdb_path is None:
            raise FileNotFoundError(f"PDB not found for peptide_id={row['peptide_id']!r}")

        aa_sequence, ca_coords, plddt_values = parse_pdb(str(pdb_path))
        n_residues = len(aa_sequence)

        x = compute_node_features(aa_sequence, plddt_values, n_residues, groups=self.node_feature_groups)
        edge_index, edge_attr = compute_edges(ca_coords, self.distance_threshold)
        pos = torch.tensor(ca_coords, dtype=torch.float32)

        label = int(row["label"])

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=n_residues,
        )

        if self.feature_cols:
            if self.tabular_scaler is not None:
                extra = self.tabular_scaler.transform_row(row).astype(np.float32)
            else:
                extra = row[self.feature_cols].values.astype(np.float32)
                extra = np.nan_to_num(extra, nan=0.0)
            data.geo_features = torch.tensor(extra, dtype=torch.float32).unsqueeze(0)

        if self.esm2_residue_dir is not None and wants_esm2_residue_nodes(self.node_feature_groups):
            esm = load_esm2_per_residue_tensor(self.esm2_residue_dir, row["peptide_id"])
            data.esm2_node = esm

        return data


def train_single_model(
    arch: str,
    feature_name: str,
    feature_cfg: dict,
    df: pd.DataFrame,
    qsar_cols,
    args,
    device: torch.device,
    out_dir: Path,
    esm2_raw_dim: int,
    node_feature_groups: NodeFeatureGroups | None = None,
):
    feature_cols = create_feature_cols(feature_cfg["use_geo"], feature_cfg["use_qsar"], qsar_cols)

    unique_labels = sorted(df["label"].unique())
    label_map = {label: 1 if int(label) == 1 else 0 for label in unique_labels}
    labels = np.array([label_map[label] for label in df["label"].values])
    require_both_classes(labels, context=f"Training {arch}/{feature_name}")
    df_mapped = df.copy()
    df_mapped["label"] = labels
    num_classes = 2

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(sss.split(np.arange(len(labels)), labels))

    tabular_scaler = None
    if feature_cols:
        tabular_scaler = ExtraFeatureRobustScaler.fit(df_mapped.iloc[train_idx], feature_cols, balance_blocks=True)

    want_esm2_nodes = wants_esm2_residue_nodes(node_feature_groups)
    esm2_dir = str(Path(args.esm2_residue_dir).resolve()) if want_esm2_nodes and args.esm2_residue_dir else None

    dataset = CustomPeptideDataset(
        df_mapped,
        args.pdb_dir,
        feature_cols if feature_cols else None,
        args.distance_threshold,
        tabular_scaler=tabular_scaler,
        esm2_residue_dir=esm2_dir,
        node_feature_groups=node_feature_groups,
    )

    from torch_geometric.loader import DataLoader

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    train_y = labels[train_idx]
    counts = np.bincount(train_y, minlength=num_classes)
    if int(counts.min()) > 0:
        n_tr = int(train_y.shape[0])
        w = np.array([n_tr / (num_classes * int(counts[c])) for c in range(num_classes)], dtype=np.float32)
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        class_weights = None

    geo_dim = len(feature_cols)
    esm2_raw = esm2_raw_dim if want_esm2_nodes else 0
    in_ch = node_input_dim(node_feature_groups)

    model = PeptideGNN(
        architecture=arch,
        in_channels=in_ch,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=num_classes,
        pooling="mean_max",
        geo_feature_dim=geo_dim,
        esm2_raw_dim=esm2_raw,
        esm2_hidden_dim=args.esm2_hidden_dim,
    )

    print(f"\n=== Training {arch.upper()} on {feature_name} ===")
    _, best_metrics = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        class_weights=class_weights,
        verbose=True,
        label_smoothing=args.label_smoothing,
        logit_penalty=args.logit_penalty,
    )

    model_dir = out_dir / f"{arch}_{feature_name.replace('+', '_plus_')}"
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_checkpoint_path(model_dir)
    torch.save(model.state_dict(), ckpt_path)

    if tabular_scaler is not None:
        save_extra_feature_scaler(tabular_scaler, str(tabular_scaler_path(model_dir)))

    platt_path = platt_scaling_path(model_dir)
    if num_classes == 2:
        margins, y_val = collect_margins_and_labels(model, val_loader, device)
        platt_payload = fit_platt(margins, y_val)
        if platt_payload is not None:
            save_platt_json(platt_path, platt_payload)
            print(f"Saved Platt calibration: {platt_path}")

    _meta_ng = node_feature_groups if node_feature_groups is not None else NodeFeatureGroups()
    save_peptide_gnn_meta(
        model_dir,
        architecture=arch,
        node_feature_groups=_meta_ng,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling="mean_max",
        geo_feature_dim=geo_dim,
        esm2_raw_dim=esm2_raw,
        esm2_hidden_dim=args.esm2_hidden_dim,
        num_classes=num_classes,
        label_map=label_map,
        tabular_feature_cols=feature_cols,
        feature_set=feature_name,
        metrics=best_metrics,
        timestamp=datetime.now().isoformat(),
    )
    print(f"Saved model directory: {model_dir}")

    return str(model_dir), best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train GNN from a labeled peptide CSV")
    parser.add_argument(
        "input",
        type=str,
        help="Labeled dataset CSV (id,label,sequence), or an existing generated/ folder",
    )
    parser.add_argument("--config", type=str, default="configs/gnn_final_train.json", help="Path to config JSON")
    parser.add_argument("--output_dir", type=str, default="results/gnn_models")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--esm2_hidden_dim", type=int, default=64)
    parser.add_argument("--presets", nargs="+", default=None, help="Post-pooling tabular presets to train")
    parser.add_argument("--force-process", action="store_true", help="Regenerate processed features before training")
    parser.add_argument("--device", type=str, default=None, help="Device for on-demand processing (default: auto)")
    args = parser.parse_args()

    csv_path, gen_dir = resolve_input_and_generated(args.input)
    if csv_path is not None and not input_has_labels(csv_path):
        print(
            f"Error: Training CSV must include a 'label' column (id,label,sequence): {csv_path}"
        )
        sys.exit(1)

    config_path = Path(__file__).resolve().parent / args.config
    with open(config_path) as f:
        config = json.load(f)

    for k, v in config.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    node_feature_groups = NodeFeatureGroups(**args.node_feature_groups)
    need_esm2 = wants_esm2_residue_nodes(node_feature_groups)
    process_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        gen_dir = ensure_processed_data(
            csv_path,
            gen_dir,
            need_esm2=need_esm2,
            require_labels=True,
            device=process_device,
            force=args.force_process,
        )
        merged_df, qsar_cols = load_data_with_features(
            gen_dir / "geometric_features.csv",
            gen_dir / "qsar12_descriptors.csv",
            source_csv=csv_path,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    args.csv_path = gen_dir / "geometric_features.csv"
    args.pdb_dir = gen_dir / "structures"
    args.qsar_csv = gen_dir / "qsar12_descriptors.csv"
    args.esm2_residue_dir = gen_dir / "esm2_per_residue"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    esm2_raw_dim = 0
    if need_esm2:
        esm2_raw_dim = infer_esm2_raw_dim(merged_df, args.esm2_residue_dir)

    feature_sets = args.feature_sets
    selected_presets = (
        args.presets
        or getattr(args, "feature_sets_default", None)
        or getattr(args, "train_feature_sets", None)
        or list(feature_sets)
    )
    unknown_presets = [name for name in selected_presets if name not in feature_sets]
    if unknown_presets:
        raise ValueError(
            f"Unknown presets {unknown_presets}; available: {list(feature_sets)}"
        )

    for preset_name in selected_presets:
        f_cfg = feature_sets[preset_name]
        for arch in args.architectures:
            model_dir, metrics = train_single_model(
                arch=arch,
                feature_name=preset_name,
                feature_cfg=f_cfg,
                df=merged_df,
                qsar_cols=qsar_cols,
                args=args,
                device=device,
                out_dir=out_dir,
                esm2_raw_dim=esm2_raw_dim,
                node_feature_groups=node_feature_groups,
            )
            print(
                f"Finished {arch}/{preset_name}: "
                f"val_auc={metrics.get('auc_roc', float('nan')):.4f} -> {model_dir}"
            )

    print(f"\nModels saved under: {out_dir}")


if __name__ == "__main__":
    main()
