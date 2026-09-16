"""
ESM Sequence Processor
Takes peptide sequences and generates:
1. ESMFold structure predictions (PDB files)
2. ESM-2 embeddings (features for downstream ML)

Compatible with:
- SVM input format (index + sequence)
- FASTA format
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional
import torch
import pandas as pd
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

STANDARD_AA_20 = frozenset("ACDEFGHIKLMNPQRSTVWY")

def canonical_standard_aa_sequence(seq: str) -> str | None:
    s = seq.replace(" ", "").replace("\t", "").upper()
    if not s:
        return None
    for c in s:
        if c not in STANDARD_AA_20:
            return None
    return s



def _sanitize_for_esm_alphabet(seq: str, valid_single: set[str]) -> str:
    u = seq.upper()
    return "".join(c if c in valid_single else "X" for c in u)


def _parse_svm_style_file(input_file):
    """Parse plain text sequence file: 'index sequence' or 'sequence' per line."""
    sequences = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(None, 1)
            if len(parts) == 2:
                idx, seq = parts
                sequences.append((idx, seq.strip()))
            elif len(parts) == 1:
                seq = parts[0]
                idx = len(sequences) + 1
                sequences.append((str(idx), seq.strip()))
    return sequences


def _parse_fasta_file(input_file):
    """Parse FASTA file into (index, sequence) tuples."""
    sequences = []
    current_id = None
    current_seq = []
    fallback_idx = 1

    with open(input_file, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id is not None and current_seq:
                    sequences.append((current_id, ''.join(current_seq)))
                header = line[1:].strip()
                header_id = header.split()[0] if header else f"seq_{fallback_idx}"
                current_id = header_id
                current_seq = []
                fallback_idx += 1
            else:
                if current_id is None:
                    current_id = f"seq_{fallback_idx}"
                    fallback_idx += 1
                current_seq.append(line)

    if current_id is not None and current_seq:
        sequences.append((current_id, ''.join(current_seq)))

    return sequences


def parse_sequence_file(input_file):
    """
    Parse sequence file and return list of (index, sequence) tuples.

    Supports:
    - SVM-style text: 'index sequence' (or bare sequence)
    - FASTA: lines starting with '>'
    """
    with open(input_file, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('>'):
                return _parse_fasta_file(input_file)
            break
    return _parse_svm_style_file(input_file)


def _safe_esm2_residue_filename(seq_index) -> str:
    s = str(seq_index).strip().replace("\\", "/")
    return s.replace("/", "_").replace(":", "_")


def esm2_residue_path(output_dir: Path | str, peptide_id) -> Path:
    return Path(output_dir) / f"{_safe_esm2_residue_filename(peptide_id)}.pt"


def resolve_esm2_residue_path(
    output_dir: Path | str, peptide_id
) -> Optional[Path]:
    path = esm2_residue_path(output_dir, peptide_id)
    return path if path.is_file() else None


def esmfold_pdb_path(output_dir: Path | str, peptide_id) -> Path:
    return Path(output_dir) / f"{_safe_esm2_residue_filename(peptide_id)}.pdb"


def resolve_esmfold_pdb_path(output_dir: Path | str, peptide_id) -> Optional[Path]:
    output_dir = Path(output_dir)
    stem = _safe_esm2_residue_filename(peptide_id)
    for path in (
        esmfold_pdb_path(output_dir, peptide_id),
        output_dir / f"structure_{stem}.pdb",
    ):
        if path.is_file():
            return path
    return None


def extract_esm2_embeddings(
    sequences,
    model_name="esm2_t33_650M_UR50D",
    device="cuda",
    per_residue_dir: Optional[Path | str] = None,
):
    """
    Extract ESM-2 embeddings for sequences
    
    Args:
        sequences: List of (index, sequence) tuples
        model_name: ESM-2 model to use
        device: 'cuda' or 'cpu'
        per_residue_dir: If set, save ``(L, D)`` per-residue tensors as
            ``{safe_id}.pt`` (dict with ``embedding`` key) for GNN node features.
    
    Returns:
        DataFrame with embeddings
    """
    import esm
    
    print(f"\n{'='*60}")
    print(f"  Loading ESM-2 Model: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    model, alphabet = esm.pretrained.__dict__[model_name]()
    batch_converter = alphabet.get_batch_converter()
    valid_single = {t for t in alphabet.all_toks if len(t) == 1}
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on: {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Extract embeddings
    all_embeddings = []
    all_indices = []
    
    print(f"\n{'='*60}")
    print(f"  Extracting Embeddings")
    print(f"{'='*60}")

    if per_residue_dir is not None:
        per_residue_dir = Path(per_residue_dir)
        per_residue_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Per-residue tensors → {per_residue_dir}")
    
    for idx, seq in tqdm(sequences, desc="Processing sequences"):
        canon = canonical_standard_aa_sequence(seq)
        if canon is None:
            raise ValueError(
                f"Sequence {idx!r} is not standard 20 AA only; "
                "filter with canonical_standard_aa_sequence before embeddings"
            )
        seq = _sanitize_for_esm_alphabet(canon, valid_single)
        # Prepare data
        data = [(idx, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        
        # Extract features
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
            
            token_representations = results["representations"][33]
            per_tok = token_representations[0, 1 : len(seq) + 1].float().cpu()
            if per_residue_dir is not None:
                torch.save(
                    {"seqIndex": str(idx), "embedding": per_tok},
                    esm2_residue_path(per_residue_dir, idx),
                )
            sequence_rep = per_tok.mean(0)
            
        all_embeddings.append(sequence_rep.cpu().numpy())
        all_indices.append(idx)
    
    # Create DataFrame
    embedding_dim = all_embeddings[0].shape[0]
    columns = [f"esm2_dim_{i}" for i in range(embedding_dim)]
    
    df = pd.DataFrame(all_embeddings, columns=columns)
    df.insert(0, 'seqIndex', all_indices)
    
    print(f"\n✅ Extracted embeddings: {df.shape}")
    print(f"   Embedding dimension: {embedding_dim}")
    
    return df


def _load_esmfold_model(device: str):
    from transformers import EsmForProteinFolding

    local_model_path = Path(__file__).parent / "esmfold_v1_local"
    load_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"\n{'='*60}")
    print("  Loading ESMFold Model (HuggingFace)")
    print(f"{'='*60}")

    if local_model_path.exists():
        print(f"✅ Loading from local directory: {local_model_path}")
        model = EsmForProteinFolding.from_pretrained(
            str(local_model_path),
            local_files_only=True,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
        )
    else:
        print("⚠️  Local model not found, will download from HuggingFace...")
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1",
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
        )

    model = model.to(device)
    model.eval()
    print(f"✅ ESMFold loaded on: {device}")
    return model


def _pending_folds(sequences, output_dir: Path, max_length: int, skip_existing: bool):
    results = []
    pending = []
    for order, (idx, seq) in enumerate(sequences):
        canon = canonical_standard_aa_sequence(seq)
        if canon is None:
            results.append({
                "_order": order, "seqIndex": idx, "length": len(seq),
                "status": "skipped_invalid_sequence", "pdb_file": None,
            })
            continue
        if len(canon) > max_length:
            results.append({
                "_order": order, "seqIndex": idx, "length": len(canon),
                "status": "skipped_too_long", "pdb_file": None,
            })
            continue
        existing = resolve_esmfold_pdb_path(output_dir, idx) if skip_existing else None
        if existing is not None:
            results.append({
                "_order": order, "seqIndex": idx, "length": len(canon),
                "status": "success", "pdb_file": str(existing), "cached": True,
            })
        else:
            pending.append((order, idx, canon))
    return results, pending


def predict_structures_esmfold(
    sequences,
    output_dir,
    device="cuda",
    max_length=400,
    skip_existing=True,
):
    """
    Predict 3D structures using ESMFold
    
    Args:
        sequences: List of (index, sequence) tuples
        output_dir: Directory to save PDB files
        device: 'cuda' or 'cpu'
        max_length: Maximum sequence length (longer sequences need more memory)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("  Predicting Structures")
    print(f"{'='*60}")

    results, pending = _pending_folds(
        sequences, output_dir, max_length, skip_existing
    )
    cached = sum(bool(row.get("cached")) for row in results)
    if cached:
        print(f"   Reusing {cached} existing structures")

    model = _load_esmfold_model(device) if pending else None
    for order, idx, seq in tqdm(pending, desc="Folding sequences"):
        try:
            if device == "cuda":
                torch.cuda.empty_cache()

            start_time = time.time()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
                output = model.infer_pdb(seq)
            elapsed = time.time() - start_time

            pdb_file = esmfold_pdb_path(output_dir, idx)
            with pdb_file.open("w") as f:
                f.write(output)
            results.append({
                "_order": order, "seqIndex": idx, "length": len(seq),
                "status": "success", "pdb_file": str(pdb_file),
                "time_seconds": elapsed, "cached": False,
            })
        except Exception as e:
            print(f"❌ Error processing sequence {idx}: {str(e)}")
            results.append({
                "_order": order, "seqIndex": idx, "length": len(seq),
                "status": f"error: {str(e)}", "pdb_file": None,
                "cached": False,
            })

    df_summary = pd.DataFrame(sorted(results, key=lambda row: row["_order"]))
    df_summary = df_summary.drop(columns=["_order"])
    print(f"\n✅ Structure prediction complete")
    print(f"   Success: {sum(df_summary['status'] == 'success')}/{len(sequences)}")
    print(f"   Output directory: {output_dir}")
    return df_summary


def main():
    parser = argparse.ArgumentParser(
        description="Process peptide sequences with ESM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract ESM-2 embeddings for ML
  python esm_sequence_processor.py --input seqs.txt --output embeddings.csv --mode embeddings
  
  # Predict structures with ESMFold
  python esm_sequence_processor.py --input seqs.txt --output structures/ --mode fold
  
  # Do both
  python esm_sequence_processor.py --input seqs.txt --output results/ --mode both

Input formats:
  SVM style:
    1 MKTAYIAKQRQISFVKSHFSRQL
    2 GVVDSDDLPLVVAASNAGKSTVVQLLAAAG

  FASTA:
    >seq1
    MKTAYIAKQRQISFVKSHFSRQL
    >seq2
    GVVDSDDLPLVVAASNAGKSTVVQLLAAAG
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Input sequence file (SVM-style text or FASTA)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output path (CSV for embeddings, directory for structures)')
    parser.add_argument('--mode', '-m', choices=['embeddings', 'fold', 'both'],
                        default='embeddings',
                        help='Processing mode: embeddings (ESM-2), fold (ESMFold), or both')
    parser.add_argument('--device', '-d', choices=['cuda', 'cpu'],
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available)')
    parser.add_argument('--max-length', type=int, default=400,
                        help='Maximum sequence length for folding (default: 400)')
    parser.add_argument('--esm-model', default='esm2_t33_650M_UR50D',
                        choices=['esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D', 'esm2_t30_150M_UR50D'],
                        help='ESM-2 model to use for embeddings')
    parser.add_argument(
        '--per-residue-dir',
        type=str,
        default=None,
        help=(
            'If set (embeddings mode), also write per-residue ESM-2 layer tensors '
            'as {seqIndex}.pt files under this directory for GNN node features.'
        ),
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "🧬" * 30)
    print("   ESM Sequence Processor")
    print("🧬" * 30)
    
    # Check GPU
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("❌ CUDA requested but not available. Falling back to CPU.")
            args.device = 'cpu'
        else:
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("ℹ️  Using CPU (this will be slower)")
    
    # Load sequences
    print(f"\n{'='*60}")
    print(f"  Loading Sequences")
    print(f"{'='*60}")
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    raw_sequences = parse_sequence_file(args.input)
    sequences = []
    n_skipped_invalid = 0
    for idx, seq in raw_sequences:
        canon = canonical_standard_aa_sequence(seq)
        if canon is None:
            n_skipped_invalid += 1
            continue
        sequences.append((idx, canon))
    if n_skipped_invalid:
        print(
            f"   Skipped {n_skipped_invalid} sequences (non-standard letters or X; "
            "only standard 20 amino acids accepted)"
        )
    if not sequences:
        print("❌ No valid sequences after filtering.", file=sys.stderr)
        sys.exit(1)
    print(f"✅ Loaded {len(sequences)} sequences")
    print(f"   Length range: {min(len(s[1]) for s in sequences)} - {max(len(s[1]) for s in sequences)} aa")
    
    # Process based on mode
    if args.mode in ['embeddings', 'both']:
        # Extract embeddings
        pr_dir = Path(args.per_residue_dir).resolve() if args.per_residue_dir else None
        embeddings_df = extract_esm2_embeddings(
            sequences,
            model_name=args.esm_model,
            device=args.device,
            per_residue_dir=pr_dir,
        )
        
        # Save embeddings
        if args.mode == 'embeddings':
            output_file = args.output
        else:
            output_file = os.path.join(args.output, 'esm2_embeddings.csv')
            os.makedirs(args.output, exist_ok=True)
        
        embeddings_df.to_csv(output_file, index=False)
        print(f"\n💾 Embeddings saved to: {output_file}")
    
    if args.mode in ['fold', 'both']:
        # Predict structures
        if args.mode == 'fold':
            output_dir = args.output
        else:
            output_dir = os.path.join(args.output, 'structures')
        
        summary_df = predict_structures_esmfold(
            sequences,
            output_dir=output_dir,
            device=args.device,
            max_length=args.max_length
        )
        n_ok = int((summary_df["status"] == "success").sum()) if not summary_df.empty else 0
        print(f"\n💾 Structures saved to: {output_dir} ({n_ok}/{len(summary_df)} successful)")
    
    # Final summary
    print(f"\n{'='*60}")
    print("  ✅ Processing Complete!")
    print(f"{'='*60}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Mode:   {args.mode}")
    print(f"Device: {args.device}")
    print()


if __name__ == "__main__":
    main()
