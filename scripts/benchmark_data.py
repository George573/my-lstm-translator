#!/usr/bin/env python3
"""Measure the real coverage sampler and CSV input pipeline without training."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
import statistics
import time

import torch
from torch.utils.data import DataLoader

from lstm_translator import BPETokenizer, IndexedTranslationDataset, Vocabulary
from lstm_translator.indexed import CoverageSampler
from lstm_translator.model import collate_translation_batch

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def measure_loader(dataset, indices, batch_size, workers, warmup):
    loader = DataLoader(
        dataset, sampler=indices, batch_size=batch_size, num_workers=workers,
        collate_fn=collate_translation_batch, pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(42),
    )
    started = time.perf_counter()
    iterator = iter(loader)
    startup = time.perf_counter() - started
    waits = []
    examples = tokens = padded = 0
    max_source = max_target = 0
    for batch_number in range(len(loader)):
        started = time.perf_counter()
        source, target, lengths = next(iterator)
        elapsed = time.perf_counter() - started
        if batch_number == 0:
            print(f"  worker startup + first batch: {startup + elapsed:.3f}s", flush=True)
        if batch_number < warmup:
            continue
        waits.append(elapsed)
        examples += source.size(0)
        tokens += int(lengths.sum()) + int(target.ne(0).sum())
        padded += source.numel() + target.numel()
        max_source = max(max_source, source.size(1))
        max_target = max(max_target, target.size(1))
    # Exhaust the iterator so worker processes shut down normally.
    next(iterator, None)
    if not waits:
        raise ValueError("not enough batches after warmup; reduce --warmup or --batch-size")
    ordered = sorted(waits)
    print(
        f"  {len(waits)} measured batches; {len(waits) / sum(waits):.2f} batches/s; "
        f"{examples / sum(waits):,.0f} pairs/s\n"
        f"  next-batch wait: mean {statistics.mean(waits) * 1000:.2f}ms, "
        f"median {statistics.median(waits) * 1000:.2f}ms, "
        f"p95 {ordered[min(len(ordered) - 1, int(len(ordered) * .95))] * 1000:.2f}ms\n"
        f"  max padded lengths: source={max_source}, target={max_target}; "
        f"padding fraction={1 - tokens / padded:.1%}", flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--source-tokenizer", type=Path, default=ROOT / "artifacts/tokenizers/en.json")
    parser.add_argument("--target-tokenizer", type=Path, default=ROOT / "artifacts/tokenizers/fr.json")
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--batches", type=positive_int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.warmup < 0 or any(worker < 0 for worker in args.workers):
        parser.error("warmup and workers must be non-negative")

    source = BPETokenizer.load(args.source_tokenizer)
    target = BPETokenizer.load(args.target_tokenizer)
    print(f"Dataset: {args.data.resolve()}\nBuilding/loading record index...", flush=True)
    started = time.perf_counter()
    dataset = IndexedTranslationDataset(
        args.data, source, target, Vocabulary(source.tokens), Vocabulary(target.tokens),
        seed=args.seed, index_dir=args.index_dir,
    )
    print(f"Index setup: {time.perf_counter() - started:.3f}s; {len(dataset):,} training pairs", flush=True)
    started = time.perf_counter()
    sampler = CoverageSampler(len(dataset), args.seed)
    print(f"Full permutation setup: {time.perf_counter() - started:.3f}s "
          f"({len(sampler.order) * sampler.order.itemsize / 2**20:.1f} MiB)", flush=True)
    count = min(len(dataset), (args.batches + args.warmup) * args.batch_size)
    started = time.perf_counter()
    indices = list(islice(sampler, count))
    elapsed = time.perf_counter() - started
    print(f"Sampler iteration: {elapsed:.6f}s for {count:,} indices "
          f"({elapsed / count * args.batch_size * 1000:.4f}ms per full batch)", flush=True)

    # Compare identical records, changing only their physical read order.
    for label, order in [("random", indices), ("offset-sorted", sorted(indices))]:
        started = time.perf_counter()
        for index in order:
            dataset.raw_pair(index)
        elapsed = time.perf_counter() - started
        print(f"CSV read + parse, {label}: {elapsed:.3f}s; {count / elapsed:,.0f} pairs/s", flush=True)

    pairs = [dataset.raw_pair(index) for index in indices[:min(count, 4096)]]
    source._encode_cache.clear()
    target._encode_cache.clear()
    started = time.perf_counter()
    for english, french in pairs:
        source.encode(english)
        target.encode(french)
    elapsed = time.perf_counter() - started
    print(f"BPE only, {len(pairs):,} in-memory pairs: {elapsed:.3f}s; "
          f"{len(pairs) / elapsed:,.0f} pairs/s", flush=True)
    del pairs

    for workers in args.workers:
        source._encode_cache.clear()
        target._encode_cache.clear()
        dataset.close()
        print(f"Full DataLoader, random order, workers={workers}:", flush=True)
        measure_loader(dataset, indices, args.batch_size, workers, args.warmup)
    dataset.close()
    print("Read-only benchmark except for record-index cache creation. No model or checkpoints loaded.\n"
          "Later stages may benefit from filesystem caching; this is not a cold-cache test.\n"
          "Loader timings exclude GPU work and do not measure overlap during training.\n"
          "Compare the same CSV and options on each mount; 1.8 training batches/s = 556ms/batch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
