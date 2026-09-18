"""Command-line evaluation shared by current and historical experiment scripts."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from sacrebleu.metrics import BLEU, CHRF

from .checkpoint import load_checkpoint
from .cli import fraction, positive_int
from .data import iter_parallel_rows, parallel_partition
from .model import EOS_INDEX, PAD_INDEX, SOS_INDEX


def sample_pairs(path, size, seed, test_fraction, scheme):
    """Keep historical pair hashing explicit; new evaluations group by source."""
    candidates = []
    seen = set()
    for pair in iter_parallel_rows(path):
        if scheme == "legacy-pair":
            key = f"{seed}\0{pair[0]}\0{pair[1]}".encode("utf-8")
            selected = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2**64 < test_fraction
        else:
            selected = parallel_partition(*pair, validation_fraction=0,
                                          test_fraction=test_fraction, seed=seed) == "test"
        if selected and pair not in seen:
            seen.add(pair)
            candidates.append(pair)
    if not candidates:
        raise ValueError(f"no test examples in {path}")
    return random.Random(seed).sample(candidates, min(size, len(candidates)))


def evaluate_bundle(bundle, pairs, batch_size, max_length):
    started = time.monotonic()
    hypotheses, capped = [], 0
    for offset in range(0, len(pairs), batch_size):
        batch = pairs[offset:offset + batch_size]
        tensors = [torch.tensor(bundle.source_vocabulary.encode(
            bundle.source_tokenizer.encode(source)) + [EOS_INDEX]) for source, _ in batch]
        with torch.inference_mode():
            generated = bundle.model.generate(
                pad_sequence(tensors, batch_first=True, padding_value=PAD_INDEX),
                torch.tensor([len(tensor) for tensor in tensors]), max_length=max_length,
            )
        for row in generated.tolist():
            capped += EOS_INDEX not in row
            tokens = [index for index in row if index not in {EOS_INDEX, PAD_INDEX, SOS_INDEX}]
            hypotheses.append(bundle.target_tokenizer.decode(bundle.target_vocabulary.decode(tokens)))
    references = [target for _, target in pairs]
    bleu, chrf = BLEU(), CHRF()
    return {
        "bleu": bleu.corpus_score(hypotheses, [references]).score,
        "chrf": chrf.corpus_score(hypotheses, [references]).score,
        "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature()),
        "capped": capped, "seconds": time.monotonic() - started,
        "examples": [{"source": source, "reference": target, "hypothesis": hypothesis}
                     for (source, target), hypothesis in zip(pairs, hypotheses)],
    }


def main(argv=None, *, defaults=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--data", type=Path, nargs="+")
    inputs.add_argument("--examples-from", type=Path, help="reuse exact examples from a saved evaluation")
    parser.add_argument("--reference-model", help="model key in --examples-from (defaults to first model)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-size", type=positive_int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=fraction, default=0.01)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--max-length", type=positive_int, default=100)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--partition-scheme", choices=("source", "legacy-pair"), default="source")
    args = parser.parse_args(argv)
    defaults = defaults or {}
    # Explicit input flags replace the historical input default as a unit.
    if args.data is None and args.examples_from is None:
        args.data = defaults.get("data")
        args.examples_from = defaults.get("examples_from")
    args.checkpoints = args.checkpoints or defaults.get("checkpoints")
    args.output = args.output or defaults.get("output")
    if not args.checkpoints or not args.output or (not args.data and not args.examples_from):
        parser.error("supply --checkpoints, --output, and --data or --examples-from")
    if len({Path(path).name for path in args.checkpoints}) != len(args.checkpoints):
        parser.error("checkpoint basenames must be unique")
    try:
        torch.set_num_threads(args.threads)
        if args.examples_from:
            previous = json.loads(Path(args.examples_from).read_text(encoding="utf-8"))
            models = previous["models"]
            model_name = args.reference_model or next(iter(models))
            datasets = {path: [(item["source"], item["reference"]) for item in scores["examples"]]
                        for path, scores in models[model_name]["datasets"].items()}
        else:
            datasets = {str(path): sample_pairs(path, args.sample_size, args.seed,
                                                args.test_fraction, args.partition_scheme)
                        for path in args.data}
        if not datasets or any(not pairs for pairs in datasets.values()):
            raise ValueError("evaluation requires nonempty datasets")
        results = {
            "seed": args.seed, "sample_per_corpus": args.sample_size,
            "max_length": args.max_length, "device": "cpu",
            "partition_scheme": "saved-examples" if args.examples_from else args.partition_scheme,
            "examples_from": str(args.examples_from) if args.examples_from else None,
            "test_fraction": args.test_fraction,
            "note": "Training overlap is not verified; use the original training split or saved examples.",
            "models": {},
        }
        for path in args.checkpoints:
            path = Path(path)
            bundle = load_checkpoint(path, "cpu")
            state = bundle.training_state or {}
            result = {"metadata": bundle.metadata,
                      "parameters": sum(parameter.numel() for parameter in bundle.model.parameters()),
                      "global_step": state.get("global_step"), "history": state.get("history", []),
                      "teacher_forcing_ratio": state.get("teacher_forcing_ratio"), "datasets": {}}
            bundle.training_state = None
            del state
            for name, pairs in datasets.items():
                result["datasets"][name] = evaluate_bundle(bundle, pairs, args.batch_size, args.max_length)
                print(path.name, name, {key: value for key, value in result["datasets"][name].items()
                                       if key != "examples"}, flush=True)
            results["models"][path.name] = result
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, StopIteration, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0
