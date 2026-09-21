#!/usr/bin/env python3
"""Measure tokenizer word extraction on a bounded corpus prefix, without merges."""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import platform
import time

from lstm_translator import BPETokenizer, TypoGenerator, iter_parallel_rows, seed_everything
from lstm_translator.typo import tokenizer_training_texts


class TimedInput:
    """Measure time spent producing input inside the extraction pipeline."""

    def __init__(self, texts):
        self.iterator = iter(texts)
        self.seconds = 0.0
        self.texts = 0

    def __iter__(self):
        return self

    def __next__(self):
        started = time.perf_counter()
        try:
            text = next(self.iterator)
            self.texts += 1
            return text
        finally:
            self.seconds += time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/csv/en-fr.csv"))
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rows, args.repeats, args.chunk_size, *args.workers) < 1:
        parser.error("rows, repeats, chunk-size and workers must be positive")

    def sources():
        return (en for en, _ in islice(iter_parallel_rows(args.data), args.rows))

    def noise():
        return TypoGenerator(corruption_probability=0.23, lang="en",
                             aug_char_p=0.1, aug_word_p=0.1,
                             aug_char_max=1, aug_word_max=3)

    records = []

    def measure(label, function, **metadata):
        started = time.perf_counter()
        result = function()
        seconds = time.perf_counter() - started
        record = dict(stage=label, seconds=seconds, **metadata)
        records.append(record)
        print(json.dumps(record), flush=True)
        return result, record

    # Separate lazy nlpaug import/backend initialization from steady-state work.
    def warmup():
        generator = noise()
        generator.corruption_probability = 1
        generator("Please translate this beautiful sentence.")

    measure("augmentation_first_call", warmup)
    for repeat in range(args.repeats):
        clean, _ = measure("csv_read", lambda: list(sources()), repeat=repeat)
        seed_everything(args.seed)
        augmented, record = measure(
            "augmentation_only", lambda: list(tokenizer_training_texts(clean, noise())),
            repeat=repeat,
        )
        record.update(source_rows=len(clean), emitted_texts=len(augmented))
        reference = None
        # Alternate order to reduce systematic warm-cache/order bias.
        workers = args.workers if repeat % 2 == 0 else args.workers[::-1]
        for count in workers:
            frequencies, _ = measure(
                "count_prepared_texts",
                lambda: BPETokenizer()._extract_words(iter(augmented), count, args.chunk_size),
                repeat=repeat, workers=count,
            )
            if reference is None:
                reference = frequencies
            assert frequencies == reference, "worker counts differ"
            seed_everything(args.seed)
            timed = TimedInput(tokenizer_training_texts(sources(), noise()))
            frequencies, record = measure(
                "streaming_total",
                lambda: BPETokenizer()._extract_words(timed, count, args.chunk_size),
                repeat=repeat, workers=count,
            )
            assert frequencies == reference, "streaming and prepared counts differ"
            record.update(input_seconds=timed.seconds, emitted_texts=timed.texts,
                          unique_words=len(frequencies), source_rows=len(clean))
            print(f"  input generation: {timed.seconds:.3f}s; "
                  f"{len(clean) / record['seconds']:,.0f} source rows/s", flush=True)

    report = dict(python=platform.python_version(), data=str(args.data.resolve()),
                  corpus_bytes=args.data.stat().st_size, rows_requested=args.rows,
                  chunk_size=args.chunk_size, seed=args.seed, records=records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
