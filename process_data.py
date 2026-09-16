#!/usr/bin/env python3
"""
Process peptide CSVs into GNN artifacts (structures, ESM2, QSAR, geometric features).

Library module used by train.py and inference.py. Not a CLI entry-point.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from pipeline.esm2_processor import (
    canonical_standard_aa_sequence,
    esm2_residue_path,
    esmfold_pdb_path,
    extract_esm2_embeddings,
    predict_structures_esmfold,
    resolve_esm2_residue_path,
    resolve_esmfold_pdb_path,
)
from pipeline.qsar import QSAR_COLUMNS, compute_qsar12
from pipeline.features import extract_all_features, get_feature_names
from pipeline.resume import (
    discover_generated_dirs,
    load_cached_table,
    matching_cached_rows,
    reuse_matching_files,
)


def default_generated_dir(input_path: Path | str) -> Path:
    return Path(input_path).resolve().parent / "generated"


def artifacts_ready(gen_dir: Path | str, *, need_esm2: bool = False) -> bool:
    gen_dir = Path(gen_dir)
    if not (gen_dir / "geometric_features.csv").is_file():
        return False
    if not (gen_dir / "qsar12_descriptors.csv").is_file():
        return False
    if not (gen_dir / "structures").is_dir():
        return False
    if need_esm2 and not (gen_dir / "esm2_per_residue").is_dir():
        return False
    return True


def input_has_labels(input_path: Path | str) -> bool:
    return "label" in pd.read_csv(input_path, nrows=0).columns


def resolve_input_and_generated(path: Path | str) -> tuple[Path | None, Path]:
    """Accept a dataset CSV or an existing generated/ directory."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_dir():
        return None, path
    if path.is_file():
        return path, default_generated_dir(path)
    raise ValueError(f"Expected a CSV file or generated/ directory, got: {path}")


def ensure_processed_data(
    input_path: Path | str | None,
    gen_dir: Path | str,
    *,
    need_esm2: bool = True,
    require_labels: bool = False,
    device: str | None = None,
    force: bool = False,
) -> Path:
    """
    Ensure ``gen_dir`` has the artifacts needed for train/inference.

    If artifacts are missing (or ``force``), ``input_path`` must be a CSV and
    processing is run automatically.
    """
    gen_dir = Path(gen_dir).resolve()
    if artifacts_ready(gen_dir, need_esm2=need_esm2) and not force:
        return gen_dir

    if input_path is None:
        missing = "complete generated artifacts"
        if need_esm2:
            missing += " including esm2_per_residue/"
        raise FileNotFoundError(
            f"Missing {missing} under {gen_dir}. "
            "Pass a source CSV so processing can run automatically."
        )

    input_path = Path(input_path).resolve()
    print("Processed features missing or incomplete; running data processing...")
    return process_dataset(
        input_path,
        device=device,
        skip_esm2=not need_esm2,
        force=force,
        output_dir=gen_dir,
        require_labels=require_labels,
    )


def prepare_input(
    df: pd.DataFrame,
    *,
    require_labels: bool = False,
) -> tuple[pd.DataFrame, bool]:
    required_cols = {"id", "sequence"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Input CSV must contain {sorted(required_cols)}; found {sorted(df.columns)}"
        )
    out = df.copy()
    out["id"] = out["id"].astype(str).str.strip()
    if out["id"].duplicated().any():
        duplicates = out.loc[out["id"].duplicated(), "id"].unique().tolist()
        raise ValueError(f"Input IDs must be unique; duplicates include {duplicates[:10]}")

    has_labels = "label" in out.columns
    if require_labels and not has_labels:
        raise ValueError("Input CSV must contain a 'label' column for training")
    if has_labels and out["label"].isna().any():
        raise ValueError("Input label column contains missing values")

    out["_canonical_sequence"] = [
        canonical_standard_aa_sequence(str(seq)) for seq in out["sequence"]
    ]
    return out, has_labels


def select_folded_records(df: pd.DataFrame, fold_summary: pd.DataFrame) -> pd.DataFrame:
    successful = set(
        fold_summary.loc[fold_summary["status"] == "success", "seqIndex"].astype(str)
    )
    ready = df[
        df["_canonical_sequence"].notna() & df["id"].isin(successful)
    ].copy()
    ready["sequence"] = ready.pop("_canonical_sequence")
    skipped = len(df) - len(ready)
    if skipped:
        print(f"   Excluding {skipped} records without usable structures")
    return ready


def ordered_rows(expected: pd.DataFrame, *tables: pd.DataFrame) -> pd.DataFrame:
    available = [table for table in tables if not table.empty]
    if not available:
        return pd.DataFrame()
    combined = pd.concat(available, ignore_index=True)
    combined["peptide_id"] = combined["peptide_id"].astype(str)
    combined = combined.drop_duplicates("peptide_id", keep="first")
    order = expected[["id"]].rename(columns={"id": "peptide_id"})
    return order.merge(combined, on="peptide_id", how="left")


def pooled_esm2_row(peptide_id: str, tensor_path: Path) -> dict:
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    embedding = payload["embedding"] if isinstance(payload, dict) else payload
    pooled = embedding.float().mean(dim=0).numpy()
    row = {"peptide_id": peptide_id, "seqIndex": peptide_id}
    row.update({f"esm2_dim_{i}": value for i, value in enumerate(pooled)})
    return row


def run_esm2_step(
    sequences,
    expected: pd.DataFrame,
    generated_dirs: list[Path],
    esm2_csv: Path,
    esm2_dir: Path,
    device: str,
    force: bool,
) -> None:
    cached = pd.DataFrame()
    if not force:
        cached = load_cached_table(
            generated_dirs, "esm2_embeddings.csv", ["peptide_id", "esm2_dim_0"]
        )
        cached = matching_cached_rows(cached, expected)

    missing = list(sequences) if force else [
        (peptide_id, sequence)
        for peptide_id, sequence in sequences
        if resolve_esm2_residue_path(esm2_dir, peptide_id) is None
    ]
    computed = pd.DataFrame()
    if missing:
        computed = extract_esm2_embeddings(
            missing, per_residue_dir=esm2_dir, device=device
        )
        computed.insert(0, "peptide_id", computed["seqIndex"].astype(str))

    pooled_ids = set(cached.get("peptide_id", pd.Series(dtype=str)).astype(str))
    pooled_ids.update(computed.get("peptide_id", pd.Series(dtype=str)).astype(str))
    rebuilt = pd.DataFrame([
        pooled_esm2_row(peptide_id, esm2_residue_path(esm2_dir, peptide_id))
        for peptide_id, _ in sequences
        if peptide_id not in pooled_ids
    ])
    embeddings = ordered_rows(expected, cached, computed, rebuilt)
    embeddings.to_csv(esm2_csv, index=False)
    print(f"   Reused {len(sequences) - len(missing)} per-residue ESM2 tensors")
    print(f"Saved pooled ESM2 embeddings to {esm2_csv}")


def build_qsar_features(
    expected: pd.DataFrame,
    generated_dirs: list[Path],
    force: bool,
) -> pd.DataFrame:
    cached = pd.DataFrame()
    if not force:
        cached = load_cached_table(
            generated_dirs,
            "qsar12_descriptors.csv",
            ["peptide_id", "sequence", *QSAR_COLUMNS],
        )
        cached = matching_cached_rows(cached, expected)
    cached_ids = set(cached.get("peptide_id", pd.Series(dtype=str)).astype(str))
    missing = expected[~expected["id"].isin(cached_ids)]
    computed = compute_qsar12(
        missing["sequence"].tolist(), missing["id"].tolist()
    )
    print(f"   Reused {len(cached)} cached QSAR rows")
    return ordered_rows(expected, cached, computed)


def _geo_id_cols(has_labels: bool) -> list[str]:
    cols = ["peptide_id", "sequence", "pdb_file"]
    if has_labels:
        cols.append("label")
    return cols


def build_geometric_features(
    sequences,
    labels: dict | None,
    pdb_dir: Path,
    *,
    has_labels: bool,
) -> pd.DataFrame:
    rows = []
    for peptide_id, sequence in sequences:
        pdb_path = resolve_esmfold_pdb_path(pdb_dir, peptide_id)
        if pdb_path is None:
            print(f"Warning: PDB not found for {peptide_id}, skipping.")
            continue
        try:
            features = extract_all_features(
                pdb_path=str(pdb_path),
                peptide_id=peptide_id,
                sequence=sequence,
                svm_sigma=None,
                svm_prob=None,
                qsar_descriptors=None,
            )
            if has_labels:
                features["label"] = labels[peptide_id]
            features["pdb_file"] = pdb_path.name
            rows.append(features)
        except Exception as exc:
            print(f"Error extracting features for {peptide_id}: {exc}")

    if not rows:
        raise RuntimeError("No geometric features were produced from the folded structures")
    geo_df = pd.DataFrame(rows)
    id_cols = _geo_id_cols(has_labels)
    feature_cols = get_feature_names(include_optional=True)
    front = [c for c in id_cols + feature_cols if c in geo_df.columns]
    remainder = [c for c in geo_df.columns if c not in front]
    return geo_df[front + remainder]


def build_or_reuse_geometric_features(
    expected: pd.DataFrame,
    generated_dirs: list[Path],
    pdb_dir: Path,
    force: bool,
    *,
    has_labels: bool,
) -> pd.DataFrame:
    cached = pd.DataFrame()
    required = ["peptide_id", "sequence", "pdb_file", *get_feature_names(False)]
    if not force:
        cached = load_cached_table(generated_dirs, "geometric_features.csv", required)
        cached = matching_cached_rows(cached, expected)
        if not cached.empty:
            cached["_resolved_pdb"] = [
                resolve_esmfold_pdb_path(pdb_dir, peptide_id)
                for peptide_id in cached["peptide_id"]
            ]
            cached = cached[cached["_resolved_pdb"].notna()].copy()
            cached["pdb_file"] = cached.pop("_resolved_pdb").map(lambda path: path.name)
            if has_labels:
                labels = dict(zip(expected["id"], expected["label"]))
                cached["label"] = cached["peptide_id"].map(labels)
            elif "label" in cached.columns:
                cached = cached.drop(columns=["label"])

    cached_ids = set(cached.get("peptide_id", pd.Series(dtype=str)).astype(str))
    missing = expected[~expected["id"].isin(cached_ids)]
    computed = pd.DataFrame()
    if not missing.empty:
        sequences = list(zip(missing["id"], missing["sequence"]))
        labels = dict(zip(missing["id"], missing["label"])) if has_labels else None
        computed = build_geometric_features(
            sequences, labels, pdb_dir, has_labels=has_labels
        )
    print(f"   Reused {len(cached)} cached geometric rows")
    return ordered_rows(expected, cached, computed)


def process_dataset(
    input_path: Path | str,
    *,
    device: str | None = None,
    skip_esm2: bool = False,
    force: bool = False,
    output_dir: Path | str | None = None,
    require_labels: bool = False,
) -> Path:
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df, has_labels = prepare_input(
        pd.read_csv(input_path), require_labels=require_labels
    )
    if not has_labels:
        print("   No label column found; writing unlabeled geometric features")

    out_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else default_generated_dir(input_path)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = out_dir / "structures"
    pdb_dir.mkdir(exist_ok=True)
    esm2_dir = out_dir / "esm2_per_residue"
    esm2_dir.mkdir(exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    esm2_csv = out_dir / "esm2_embeddings.csv"
    qsar_csv = out_dir / "qsar12_descriptors.csv"
    geo_csv = out_dir / "geometric_features.csv"

    all_sequences = list(zip(df["id"], df["sequence"].astype(str)))
    generated_dirs = (
        [out_dir.resolve()]
        if force
        else discover_generated_dirs(input_path, out_dir)
    )
    if len(generated_dirs) > 1:
        print("Resume search directories:")
        for directory in generated_dirs:
            print(f"   {directory}")

    if not force:
        reused = reuse_matching_files(
            df["id"],
            pdb_dir,
            generated_dirs,
            "structures",
            resolve_esmfold_pdb_path,
            esmfold_pdb_path,
        )
        if reused:
            print(f"   Linked or copied {reused} cached structures into {pdb_dir}")

    print("\n" + "=" * 60)
    print(f"Processing {len(all_sequences)} sequences from {input_path.name}")
    print("=" * 60)

    print("\n1. Running ESMFold...")
    fold_summary = predict_structures_esmfold(
        all_sequences,
        output_dir=pdb_dir,
        device=device,
        skip_existing=not force,
    )

    ready_df = select_folded_records(df, fold_summary)
    if ready_df.empty:
        raise RuntimeError("No records have usable structures")
    sequences = list(zip(ready_df["id"], ready_df["sequence"]))
    peptide_ids = ready_df["id"].tolist()

    print("\n2. Running ESM2..." if not skip_esm2 else "\n2. Skipping optional ESM2.")
    if not skip_esm2:
        if not force:
            reused = reuse_matching_files(
                peptide_ids,
                esm2_dir,
                generated_dirs,
                "esm2_per_residue",
                resolve_esm2_residue_path,
                esm2_residue_path,
            )
            if reused:
                print(f"   Linked or copied {reused} cached ESM2 tensors")
        run_esm2_step(
            sequences,
            ready_df,
            generated_dirs,
            esm2_csv,
            esm2_dir,
            device,
            force,
        )

    print("\n3. Computing QSAR-12 descriptors...")
    qsar_df = build_qsar_features(ready_df, generated_dirs, force)
    qsar_df.to_csv(qsar_csv, index=False)
    print(f"Saved QSAR descriptors to {qsar_csv}")

    print("\n4. Extracting Geometric Features...")
    geo_df = build_or_reuse_geometric_features(
        ready_df, generated_dirs, pdb_dir, force, has_labels=has_labels
    )
    geo_df.to_csv(geo_csv, index=False)
    print(f"Saved geometric features to {geo_csv}")

    print("\n" + "=" * 60)
    print(f"✅ Data processing complete! Outputs saved to: {out_dir}")
    print("=" * 60)
    return out_dir
