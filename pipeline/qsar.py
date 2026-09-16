#!/usr/bin/env python3
"""
Generate QSAR-12 descriptors using the same GRAR740104 path as the Lee-style SVM.

Sequence-order descriptors (tau2/tau4/QSO50/QSO29) are computed via GetSOCNp /
GetQSOp with the AAIndex GRAR740104 matrix — they are not nulled or zero-filled
on failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

QSAR_COLUMNS = [
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

CHARGE_DICT = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0, "G": 0, "H": 1, "I": 0,
    "K": 1, "L": 0, "M": 0, "N": 0, "P": 0, "Q": 0, "R": 1, "S": 0,
    "T": 0, "V": 0, "W": 0, "Y": 0,
}


def _aaindex_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        here.parents[1] / "descriptors" / "aaindex",
        here.parents[2] / "descriptors" / "aaindex",
    ]


def load_grar740104_matrix():
    try:
        from propy import AAIndex
    except Exception as error:
        raise RuntimeError(
            "Could not import ProPy. Install propy3 before computing QSAR-12."
        ) from error

    last_error: Exception | None = None
    for aaindex_dir in _aaindex_candidates():
        if not aaindex_dir.is_dir():
            continue
        try:
            return AAIndex.GetAAIndex23("GRAR740104", path=str(aaindex_dir))
        except Exception as error:
            last_error = error
    searched = ", ".join(str(path) for path in _aaindex_candidates())
    raise RuntimeError(
        f"Could not load GRAR740104 AAIndex; searched: {searched}"
    ) from last_error


def compute_sequence_order_descriptors(descriptor, sequence: str, grar740104_matrix) -> dict[str, float]:
    try:
        socn = descriptor.GetSOCNp(maxlag=30, distancematrix=grar740104_matrix)
        qso = descriptor.GetQSOp(maxlag=30, weight=0.05, distancematrix=grar740104_matrix)
    except Exception as error:
        raise RuntimeError(
            f"Could not compute GRAR740104 descriptors for sequence {sequence!r} "
            f"(length {len(sequence)})."
        ) from error

    required = {"tau2": socn, "tau4": socn, "QSO50": qso, "QSO29": qso}
    missing = [name for name, values in required.items() if name not in values]
    if missing:
        raise RuntimeError(
            f"ProPy omitted required descriptors for sequence {sequence!r}: {', '.join(missing)}."
        )

    length = len(sequence)
    return {
        "tau2_GRAR740104": float(socn["tau2"] / (length - 2)) if length > 2 else 0.0,
        "tau4_GRAR740104": float(socn["tau4"] / (length - 4)) if length > 4 else 0.0,
        "QSO50_GRAR740104": float(qso["QSO50"]),
        "QSO29_GRAR740104": float(qso["QSO29"]),
    }


def _compute_one(seq: str, pid: str, grar740104_matrix) -> dict:
    from propy.PyPro import GetProDes
    from propy import ProCheck

    if ProCheck.ProteinCheck(seq) == 0:
        raise ValueError(f"ProPy rejected sequence: {seq}")

    descriptor = GetProDes(seq)
    dpc = descriptor.GetDPComp()
    ctd = descriptor.GetCTD()
    sequence_order = compute_sequence_order_descriptors(descriptor, seq, grar740104_matrix)

    n_m = seq.count("M")
    n_k = seq.count("K")
    row = {
        "peptide_id": pid,
        "sequence": seq,
        "netCharge": float(sum(CHARGE_DICT.get(residue, 0) for residue in seq)),
        "FC": round(dpc.get("FC", 0), 2),
        "LW": round(dpc.get("LW", 0), 2),
        "DP": round(dpc.get("DP", 0), 2),
        "NK": round(dpc.get("NK", 0), 2),
        "AE": round(dpc.get("AE", 0), 2),
        "pcMK": 0.0 if n_m == 0 else n_m / (n_m + n_k),
        "_SolventAccessibilityD1025": float(ctd.get("_SolventAccessibilityD1025", 0)),
        **sequence_order,
    }
    return row


def compute_qsar12(sequences: List[str], peptide_ids: List[str]) -> pd.DataFrame:
    grar740104_matrix = load_grar740104_matrix()
    results = []
    n = len(sequences)
    for i, (seq, pid) in enumerate(zip(sequences, peptide_ids)):
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{n}...")
        results.append(_compute_one(seq, pid, grar740104_matrix))
    return pd.DataFrame(results)


def load_input(path: Path) -> Tuple[List[str], List[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        seq_col = "sequence" if "sequence" in df.columns else df.columns[1]
        id_col = None
        for c in ("peptide_id", "name", "id", "ID"):
            if c in df.columns:
                id_col = c
                break
        if id_col is None:
            id_col = df.columns[0]
        sequences = df[seq_col].astype(str).str.strip().tolist()
        peptide_ids = df[id_col].astype(str).tolist()
        return sequences, peptide_ids

    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    ids = []
    seqs = []
    for i, line in enumerate(lines):
        if "\t" in line:
            part = line.split("\t", 1)
            ids.append(part[0].strip())
            seqs.append(part[1].strip())
        elif " " in line:
            part = line.split(None, 1)
            if len(part) == 2 and part[1] and all(c.isalpha() for c in part[1].upper() if c.isalpha()):
                ids.append(part[0].strip())
                seqs.append(part[1].strip())
            else:
                ids.append(f"seq_{i + 1}")
                seqs.append(line)
        else:
            ids.append(f"seq_{i + 1}")
            seqs.append(line)
    return seqs, ids
