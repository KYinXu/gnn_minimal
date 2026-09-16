from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


def discover_generated_dirs(input_path: Path, target_dir: Path) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [target_dir.resolve()]
    current = input_path.resolve().parent
    while current == repo_root or repo_root in current.parents:
        candidates.append((current / "generated").resolve())
        if current == repo_root:
            break
        current = current.parent

    unique = []
    for candidate in candidates:
        if candidate not in unique and (candidate == target_dir.resolve() or candidate.is_dir()):
            unique.append(candidate)
    return unique


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def reuse_matching_files(
    peptide_ids: Iterable[str],
    target_dir: Path,
    generated_dirs: list[Path],
    subdirectory: str,
    resolver: Callable[[Path, str], Path | None],
    destination_for: Callable[[Path, str], Path],
) -> int:
    reused = 0
    source_dirs = [path / subdirectory for path in generated_dirs[1:]]
    for peptide_id in peptide_ids:
        if resolver(target_dir, peptide_id) is not None:
            continue
        source = next(
            (
                resolved
                for directory in source_dirs
                if (resolved := resolver(directory, peptide_id)) is not None
            ),
            None,
        )
        if source is None:
            continue
        link_or_copy(source, destination_for(target_dir, peptide_id))
        reused += 1
    return reused


def load_cached_table(
    generated_dirs: list[Path],
    filename: str,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    tables = []
    required = set(required_columns)
    for directory in generated_dirs:
        path = directory / filename
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        if required.issubset(table.columns):
            tables.append(table)
    if not tables:
        return pd.DataFrame()
    combined = pd.concat(tables, ignore_index=True)
    combined["peptide_id"] = combined["peptide_id"].astype(str)
    return combined.drop_duplicates("peptide_id", keep="first")


def matching_cached_rows(
    cached: pd.DataFrame,
    expected: pd.DataFrame,
) -> pd.DataFrame:
    if cached.empty:
        return cached
    by_id = cached.set_index("peptide_id", drop=False)
    rows = []
    for row in expected.itertuples(index=False):
        peptide_id = str(row.id)
        if peptide_id not in by_id.index:
            continue
        candidate = by_id.loc[peptide_id]
        if isinstance(candidate, pd.DataFrame):
            candidate = candidate.iloc[0]
        if "sequence" in candidate.index:
            if str(candidate["sequence"]).strip().upper() != str(row.sequence).strip().upper():
                continue
        rows.append(candidate.to_dict())
    return pd.DataFrame(rows)
