"""Helpers for loading parallel English-French corpora."""

from pathlib import Path

import pandas as pd


def load_parallel_tsv(
    path: str | Path,
    num_samples: int | float | None = None,
) -> tuple[list[str], list[str]]:
    """Load the first two tab-separated fields from a parallel corpus.

    ``num_samples`` may be a positive row count, a fraction in ``(0, 1]``, or
    ``None`` to load every valid row.
    """
    rows: list[tuple[str, str]] = []
    with Path(path).open(encoding="utf-8") as corpus:
        for line in corpus:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                rows.append((parts[0], parts[1]))

    limit = _resolve_sample_count(len(rows), num_samples)
    selected = rows[:limit]
    return [row[0] for row in selected], [row[1] for row in selected]


def load_parallel_csv(path: str | Path) -> tuple[list[str], list[str]]:
    """Load non-empty ``en`` and ``fr`` columns from a CSV file."""
    data = pd.read_csv(path, usecols=["en", "fr"]).dropna()
    return data["en"].astype(str).tolist(), data["fr"].astype(str).tolist()


def _resolve_sample_count(total: int, value: int | float | None) -> int:
    if value is None:
        return total
    if isinstance(value, bool):
        raise TypeError("num_samples must be an integer, float, or None")
    if isinstance(value, float):
        if not 0 < value <= 1:
            raise ValueError("a fractional num_samples must be in (0, 1]")
        return int(total * value)
    if isinstance(value, int):
        if value < 0:
            raise ValueError("num_samples cannot be negative")
        return min(value, total)
    raise TypeError("num_samples must be an integer, float, or None")
