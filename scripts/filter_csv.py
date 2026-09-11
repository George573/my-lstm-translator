#!/usr/bin/env python3
"""Stream a parallel CSV, keeping pairs strictly below a BPE token limit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from tqdm import tqdm

from lstm_translator.tokenizer import BPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with en/fr columns")
    parser.add_argument("--output", type=Path, required=True, help="new CSV to create")
    parser.add_argument(
        "--max-tokens", type=positive_int, required=True,
        help="exclusive limit for each sentence's BPE tokens (excludes BOS/EOS)",
    )
    parser.add_argument("--source-tokenizer", type=Path,
                        default=PROJECT_ROOT / "artifacts/tokenizers/en.json")
    parser.add_argument("--target-tokenizer", type=Path,
                        default=PROJECT_ROOT / "artifacts/tokenizers/fr.json")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("input and output must be different files")
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        source_tokenizer = BPETokenizer.load(args.source_tokenizer)
        target_tokenizer = BPETokenizer.load(args.target_tokenizer)
        # Permit large text fields without loading the corpus into memory.
        csv.field_size_limit(sys.maxsize)
        total = kept = invalid = 0
        with args.input.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            header = next(reader, None)
            if not header or header.count("en") != 1 or header.count("fr") != 1:
                raise ValueError("CSV must contain exactly one 'en' and one 'fr' column")
            source_index, target_index = header.index("en"), header.index("fr")
            with args.output.open("x", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(header)
                for row in tqdm(reader, desc="Filtering", unit=" pairs", disable=not args.progress):
                    total += 1
                    if len(row) != len(header):
                        invalid += 1
                        continue
                    english, french = row[source_index], row[target_index]
                    if not english.strip() or not french.strip():
                        invalid += 1
                        continue
                    if (len(source_tokenizer.encode(english)) < args.max_tokens
                            and len(target_tokenizer.encode(french)) < args.max_tokens):
                        writer.writerow(row)
                        kept += 1
        print(f"Read {total:,} pairs; kept {kept:,}; "
              f"filtered {total - kept - invalid:,} by length; skipped {invalid:,} invalid rows.")
        print(f"Saved to {args.output}")
    except (OSError, ValueError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
