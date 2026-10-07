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

### Node features vs post-pooling presets

Two independent switches control inputs. `--presets` (`Graph-only`, `Geo`, `QSAR`, `Combined`) chooses tabular features concatenated **after** graph pooling. `node_feature_groups` in `configs/gnn_final_train.json` chooses per-residue features **before** message passing:

| Key | What it adds |
|-----|----------------|
| `onehot` | 20-d amino-acid one-hot on each node |
| `pdb_continuous` | pLDDT, hydrophobicity, charge, molecular weight, volume, relative position |
| `vae_table` | Fixed VAE amino-acid descriptor lookup, concatenated into `data.x` |
| `esm2_residue` | Per-residue ESM2 tensor (`data.esm2_node`, typically 1280-d), projected to `esm2_hidden_dim` (default 64) and concatenated onto the node vector |

VAE and ESM2 are toggles, not the same channel. Both can stay on. With `esm2_residue` enabled, training writes tensors under `generated/esm2_per_residue/`. Use `--force-process` to regenerate artifacts after changing groups.

**Default** (`configs/gnn_final_train.json`): one-hot plus the VAE table, no PDB continuous features, no ESM2. The default preset is `Graph-only` with architecture `gat`. `python train.py path/to/dataset.csv` uses that config.

```json
"node_feature_groups": {
  "onehot": true,
  "pdb_continuous": false,
  "vae_table": true,
  "esm2_residue": false
},
"architectures": ["gat"],
"feature_sets_default": ["Graph-only"]
```

**ESM2 instead of the VAE table** (one-hot + ESM2; PDB continuous and VAE off). Same layout via config or CLI:

```bash
# Config
python train.py path/to/dataset.csv --config configs/gnn_esm2_nodes.json --force-process

# CLI (one-hot is always on; listed blocks are enabled on top)
python train.py path/to/dataset.csv --node-features esm2_residue --force-process
```

`--node-features` overrides `node_feature_groups` from the config. Tokens: `pdb_continuous` / `pdb`, `vae_table` / `vae`, `esm2_residue` / `esm2` (comma-separated for several). Omit it to keep the config defaults.

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
