"""Helpers for loading parallel English-French corpora."""

from pathlib import Path
import csv
import hashlib
import random
from typing import Iterator, Literal, Sequence

import pandas as pd

Partition = Literal["train", "validation", "test"]


def iter_parallel_rows(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield aligned rows lazily from a TSV/TXT or ``en``/``fr`` CSV file."""
    corpus_path = Path(path)
    if corpus_path.suffix.lower() == ".csv":
        with corpus_path.open(encoding="utf-8", newline="") as corpus:
            reader = csv.DictReader(corpus)
            if reader.fieldnames is None or not {"en", "fr"}.issubset(reader.fieldnames):
                raise ValueError("CSV data must contain 'en' and 'fr' columns")
            for row in reader:
                source, target = row.get("en"), row.get("fr")
                if source and target:
                    yield source, target
        return

    with corpus_path.open(encoding="utf-8") as corpus:
        for line in corpus:
            pair = _parse_parallel_line(line)
            if pair is not None:
                yield pair


def parallel_partition(
    source: str,
    target: str,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> Partition:
    """Assign a pair to a stable partition without building an index in RAM."""
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("partition fractions cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    key = f"{seed}\0{source}\0{target}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2**64
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def load_parallel_tsv(
    path: str | Path,
    num_samples: int | float | None = None,
) -> tuple[list[str], list[str]]:
    """Load the first two tab-separated fields from a parallel corpus.

    ``num_samples`` may be a positive row count, a fraction in ``(0, 1]``, or
    ``None`` to load every valid row.
    """
    corpus_path = Path(path)
    if isinstance(num_samples, float):
        total = sum(1 for _ in iter_parallel_rows(corpus_path))
        limit = _resolve_sample_count(total, num_samples)
    else:
        limit = _resolve_sample_count(num_samples, num_samples) if num_samples is not None else None

    if limit == 0:
        return [], []

    english: list[str] = []
    french: list[str] = []
    for pair in iter_parallel_rows(corpus_path):
        english.append(pair[0])
        french.append(pair[1])
        if limit is not None and len(english) >= limit:
            break
    return english, french


def load_parallel_csv(path: str | Path) -> tuple[list[str], list[str]]:
    """Load non-empty ``en`` and ``fr`` columns from a CSV file."""
    data = pd.read_csv(path, usecols=["en", "fr"]).dropna()
    return data["en"].astype(str).tolist(), data["fr"].astype(str).tolist()


def split_parallel(
    sources: Sequence[str],
    targets: Sequence[str],
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Create a deterministic shuffled train/test split without data leakage."""
    if len(sources) != len(targets):
        raise ValueError("sources and targets must contain the same number of rows")
    if len(sources) < 2:
        raise ValueError("splitting requires at least two aligned examples")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    indices = list(range(len(sources)))
    random.Random(seed).shuffle(indices)
    test_size = max(1, int(len(indices) * test_fraction))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return (
        [sources[index] for index in train_indices],
        [targets[index] for index in train_indices],
        [sources[index] for index in test_indices],
        [targets[index] for index in test_indices],
    )


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


def _parse_parallel_line(line: str) -> tuple[str, str] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]
