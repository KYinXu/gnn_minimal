# GNN Minimal Module

Self-contained module for processing peptide sequences into graphs, training GNNs, and running inference. It does not depend on the parent repository.

## Directory Structure

```text
gnn_minimal/
├── core/                  # PyTorch GNN architecture and training logic
├── pipeline/              # Data generation (ESMFold, ESM2, QSAR, Features)
├── configs/               # Default hyperparameters
├── process_data.py        # Library: CSV -> PDBs & feature CSVs (called automatically)
├── train.py               # Entry-point: labeled CSV -> model directory
├── inference.py           # Entry-point: model directory + CSV -> predictions
└── README.md
```

## Workflow

Only `train.py` and `inference.py` are CLI entry-points. Both look for a sibling `generated/` folder next to the input CSV and run processing automatically when artifacts are missing.

### 1. Train

Training requires a labeled CSV with columns `id`, `label`, `sequence`:

```csv
id,label,sequence
seq1,1,ACDEFGHIKLMNPQRSTVWY
seq2,0,VWYPQRST
```

```bash
python train.py path/to/dataset.csv
python train.py path/to/dataset.csv --presets Graph-only Geo QSAR Combined
python train.py path/to/generated/   # reuse already-processed artifacts
```

Per-residue ESM2 is controlled by `node_feature_groups.esm2_residue` in `configs/gnn_final_train.json`. Use `--force-process` to regenerate artifacts.

**Outputs** (`results/gnn_models/run_<timestamp>/<arch>_<preset>/`):

| File | Purpose |
|------|---------|
| `gnn_model.pt` | Model weights |
| `model_summary.json` | Architecture metadata + val metrics (replaces old `*_gnn_meta.json` / run summary) |
| `gnn_platt_scaling` | Probability calibration |
| `tabular_scaler.joblib` | Tabular feature scaler (Geo / QSAR / Combined only) |

### 2. Inference

Inference accepts labeled or unlabeled CSVs (`id,sequence`; `label` optional):

```csv
id,sequence
seq1,ACDEFGHIKLMNPQRSTVWY
seq2,VWYPQRST
```

```bash
python inference.py \
  --model_path results/gnn_models/run_.../gat_QSAR \
  --input path/to/new_dataset.csv
```

`--model_path` may be the model directory or `gnn_model.pt` inside it. If the input CSV has no adjacent `generated/` artifacts, processing runs first (no labels required). Predictions go to `predictions.csv` (`id`, `predicted_class`, `prob_AMP`; `true_label` only when the input CSV had labels).

### Generated artifacts

Written to `<csv_parent>/generated/`:

- `structures/` — ESMFold PDBs
- `esm2_per_residue/` — optional `(length, 1280)` tensors
- `esm2_embeddings.csv` — optional pooled ESM2
- `qsar12_descriptors.csv` — QSAR-12 via GRAR740104 `GetSOCNp`/`GetQSOp` (same as Lee-style SVM; last four not nulled)
- `geometric_features.csv` — includes `label` only when the source CSV was labeled
