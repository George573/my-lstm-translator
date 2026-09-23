#!/usr/bin/env python3
"""Translate English input interactively using a saved checkpoint."""

import argparse
import math
from pathlib import Path
import sys

import torch

from lstm_translator import Translator, load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "artifacts/checkpoints/model-32mil-param-quick-tf-decay.latest.pth",
        help="checkpoint to load (defaults to the quick-TF-decay model)",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--max-length", type=int, default=100, help="maximum generated tokens")
    parser.add_argument("--sample", action="store_true", help="sample tokens instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="positive sampling temperature (used with --sample; default: 1.0)")
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")

    try:
        torch.set_num_threads(4)
        print(f"Loading {args.checkpoint.name}...", flush=True)
        bundle = load_checkpoint(args.checkpoint, device="cpu")
        # Optimizer state and saved best weights are unnecessary for inference.
        bundle.training_state = None
        translator = Translator(bundle, device=args.device)
        print("Enter English text. Type /quit to exit (or press Ctrl+C).")
        while True:
            text = input("English text: ").strip()
            if text == "/quit":
                return 0
            if not text:
                continue
            with torch.inference_mode():
                translation = translator.translate(
                    text, max_length=args.max_length,
                    do_sample=args.sample, temperature=args.temperature,
                )
            print(f"French translation: {translation}", flush=True)
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
