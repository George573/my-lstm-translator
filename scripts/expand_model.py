#!/usr/bin/env python3
"""Append encoder and decoder LSTM layers, retaining all existing weights."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch

from lstm_translator import Seq2Seq, load_checkpoint, save_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new checkpoint path (must not already exist)")
    depth = parser.add_mutually_exclusive_group(required=True)
    depth.add_argument("--add-layers", type=int, help="number of layers to append")
    depth.add_argument("--layers", type=int, help="target total layer count")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for initializing the added layers")
    args = parser.parse_args(argv)
    if args.add_layers is not None and args.add_layers < 1:
        parser.error("--add-layers must be positive")
    try:
        if args.output.resolve() == args.checkpoint.resolve() or args.output.exists():
            raise ValueError("output must be a new path; existing checkpoints are never overwritten")
        bundle = load_checkpoint(args.checkpoint, "cpu")
        old_depth = bundle.model.config.num_layers
        new_depth = args.layers if args.layers is not None else old_depth + args.add_layers
        if new_depth <= old_depth:
            raise ValueError(f"target depth must exceed the existing {old_depth} layers")
        # The optimizer, best weights and training counters belong to the old model.
        bundle.training_state = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed)
            expanded = Seq2Seq(
                len(bundle.source_vocabulary), len(bundle.target_vocabulary),
                replace(bundle.model.config, num_layers=new_depth),
            )
        state = expanded.state_dict()
        for name, value in bundle.model.state_dict().items():
            if name not in state or state[name].shape != value.shape:
                raise ValueError(f"incompatible existing parameter: {name}")
            state[name] = value
        expanded.load_state_dict(state, strict=True)
        save_checkpoint(
            args.output, expanded, bundle.source_vocabulary, bundle.target_vocabulary,
            bundle.source_tokenizer, bundle.target_tokenizer,
            metadata={"expansion": {
                "source_checkpoint": str(args.checkpoint.resolve()),
                "source_metadata": bundle.metadata,
                "old_layers": old_depth, "new_layers": new_depth, "seed": args.seed,
            }},
        )
        print(f"Saved {args.output}: {old_depth} -> {new_depth} layers in both encoder and decoder.")
        print("Existing weights preserved; new layers randomly initialized. Further training is required.")
        print(f"Train with: python scripts/train.py --init-checkpoint {args.output} --checkpoint NEW_OUTPUT.pth")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
