"""Run paired bootstrap significance tests on two saved evaluation models."""
import argparse
import json
import os
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("rerun-significance.json"))
    parser.add_argument("--models", nargs=2, help="baseline and comparison model keys")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    previous_seed = os.environ.get("SACREBLEU_SEED")
    try:
        os.environ["SACREBLEU_SEED"] = str(args.seed)
        models = json.loads(args.input.read_text(encoding="utf-8"))["models"]
        names = args.models or list(models)
        if len(names) != 2:
            raise ValueError("select exactly two models with --models")
        baseline, comparison = (models[name] for name in names)
        if baseline["datasets"].keys() != comparison["datasets"].keys():
            raise ValueError("models have different evaluation datasets")
        results = {}
        for path in baseline["datasets"]:
            old, new = (model["datasets"][path]["examples"] for model in (baseline, comparison))
            if [(e["source"], e["reference"]) for e in old] != [(e["source"], e["reference"]) for e in new]:
                raise ValueError(f"unaligned evaluation examples in {path}")
            signatures, scores = PairedTest(
                [(names[0], [e["hypothesis"] for e in old]),
                 (names[1], [e["hypothesis"] for e in new])],
                {"BLEU": BLEU(), "chrF": CHRF()}, [[e["reference"] for e in old]],
                test_type="bs", n_samples=args.samples,
            )()
            results[path] = {"models": names, "signatures": {k: str(v) for k, v in signatures.items()},
                             "scores": {k: [vars(x) for x in v] for k, v in scores.items() if k != "System"}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    finally:
        if previous_seed is None:
            os.environ.pop("SACREBLEU_SEED", None)
        else:
            os.environ["SACREBLEU_SEED"] = previous_seed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
