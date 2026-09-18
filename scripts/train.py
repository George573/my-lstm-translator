#!/usr/bin/env python3
"""Train the English-to-French LSTM translator without Jupyter."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path
import sys
import time

import torch

from lstm_translator import (
    BPETokenizer,
    Seq2Seq,
    Seq2SeqConfig,
    IndexedTranslationDataset,
    TrainingConfig,
    Translator,
    Vocabulary,
    save_checkpoint,
    load_checkpoint,
    seed_everything,
    train_streaming_model,
    translation_scores,
    TypoGenerator,
)

from lstm_translator.cli import (
    fraction, non_negative_float, non_negative_int,
    positive_float, positive_int, probability,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the English-to-French LSTM translator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/raw/fra.txt")
    parser.add_argument(
        "--source-tokenizer",
        type=Path,
        default=PROJECT_ROOT / "artifacts/tokenizers/en.json",
    )
    parser.add_argument(
        "--target-tokenizer",
        type=Path,
        default=PROJECT_ROOT / "artifacts/tokenizers/fr.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts/checkpoints/model-v1.pth",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="JSON history path; defaults to CHECKPOINT.metrics.json",
    )
    parser.add_argument("--test-fraction", type=fraction, default=0.01)
    parser.add_argument("--evaluation-size", type=non_negative_int, default=100)
    parser.add_argument("--data-workers", type=non_negative_int, default=0)
    parser.add_argument("--source-typo-probability", type=probability, default=0.23,
                        help="probability of dynamically corrupting each training source sentence")
    parser.add_argument("--source-typo-backend", choices=("augly", "nlpaug", "none"), default="augly")
    parser.add_argument("--index-dir", type=Path, help="record-index cache directory")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path, help="resume a .latest training checkpoint")
    initialization.add_argument("--init-checkpoint", type=Path,
                                help="load model and tokenizers, starting fresh optimizer and training counters; model flags are ignored")
    parser.add_argument("--reset-patience", action="store_true",
                        help="reset the early-stopping counter on resume; keep the best validation loss")
    parser.add_argument("--resume-tf-decay-epochs", type=positive_int,
                        help="on resume, decay from saved teacher forcing to --teacher-forcing-end (which may change) over this many logged epochs")
    parser.add_argument("--validation-steps", type=positive_int, default=1_000)
    parser.add_argument(
        "--steps-per-epoch",
        type=non_negative_int,
        default=0,
        help="batches per validation interval; 0 consumes a full dataset pass",
    )
    parser.add_argument("--checkpoint-interval", type=positive_int, default=10_000)
    parser.add_argument("--log-interval", type=positive_int, default=100)

    model = parser.add_argument_group("model")
    model.add_argument("--hidden-size", type=positive_int, default=128)
    model.add_argument("--layers", type=positive_int, default=4)
    model.add_argument("--embedding-size", type=positive_int, default=42)
    model.add_argument("--dropout", type=probability, default=0.1)
    model.add_argument(
        "--no-attention", action="store_true", help="disable additive attention"
    )

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=positive_int, default=1,
                          help="total validation intervals, including completed intervals on resume")
    training.add_argument("--batch-size", type=positive_int, default=32)
    training.add_argument("--learning-rate", type=positive_float, default=1e-3)
    training.add_argument("--validation-fraction", type=fraction, default=0.01)
    training.add_argument("--teacher-forcing-start", type=probability, default=1.0)
    training.add_argument("--teacher-forcing-end", type=probability, default=0.1)
    training.add_argument("--teacher-forcing-decay-epochs", type=positive_int, default=25,
                          help="logged epochs over which teacher forcing decays; updated at each epoch start")
    training.add_argument("--patience", type=positive_int, default=5,
                          help="validation checks without improvement before stopping")
    training.add_argument("--min-delta", type=non_negative_float, default=0.01)
    training.add_argument("--gradient-clip", type=positive_float, default=1.0)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument(
        "--progress",
        "--progress-bar",
        dest="show_progress",
        action="store_true",
        help="show a tqdm progress bar while training",
    )
    training.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.reset_patience and args.resume is None:
        parser.error("--reset-patience requires --resume")
    if args.resume_tf_decay_epochs is not None and args.resume is None:
        parser.error("--resume-tf-decay-epochs requires --resume")
    try:
        device = resolve_device(args.device)
        _require_files(args.data)
        print(f"Device: {device}")
        print(f"Indexing corpus (or reusing cached offsets) from {args.data}")
        initial_checkpoint = args.resume or args.init_checkpoint
        restored = load_checkpoint(initial_checkpoint, "cpu") if initial_checkpoint else None
        if restored is not None:
            if args.resume and restored.training_state is None:
                raise ValueError("checkpoint has no resumable training state; use a new .latest checkpoint")
            if args.init_checkpoint:
                restored.training_state = None
            source_tokenizer, target_tokenizer = restored.source_tokenizer, restored.target_tokenizer
            source_vocabulary, target_vocabulary = restored.source_vocabulary, restored.target_vocabulary
        else:
            _require_files(args.source_tokenizer, args.target_tokenizer)
            source_tokenizer = BPETokenizer.load(args.source_tokenizer)
            target_tokenizer = BPETokenizer.load(args.target_tokenizer)
            source_vocabulary = Vocabulary(source_tokenizer.tokens)
            target_vocabulary = Vocabulary(target_tokenizer.tokens)
        dataset_arguments = {
            "path": args.data,
            "source_tokenizer": source_tokenizer,
            "target_tokenizer": target_tokenizer,
            "source_vocabulary": source_vocabulary,
            "target_vocabulary": target_vocabulary,
            "validation_fraction": args.validation_fraction,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
            "index_dir": args.index_dir,
        }
        training_data = IndexedTranslationDataset(
            **dataset_arguments,
            partition="train",
            source_typo_generator=(TypoGenerator(
                backend=args.source_typo_backend,
                corruption_probability=args.source_typo_probability,
            ) if args.source_typo_probability and args.source_typo_backend != "none" else None),
        )
        validation_data = IndexedTranslationDataset(
            **dataset_arguments, partition="validation"
        )
        model_config = Seq2SeqConfig(
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            embedding_dim=args.embedding_size,
            dropout=args.dropout,
            attention=not args.no_attention,
        )
        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            teacher_forcing_start=args.teacher_forcing_start,
            teacher_forcing_end=args.teacher_forcing_end,
            teacher_forcing_decay_epochs=args.teacher_forcing_decay_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            gradient_clip=args.gradient_clip,
            seed=args.seed,
            show_progress=args.show_progress,
        )
        seed_everything(args.seed)
        model = restored.model.to(device) if restored else Seq2Seq(
            len(source_vocabulary), len(target_vocabulary), model_config
        ).to(device)
        model_config = model.config
        print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

        started_at = time.monotonic()

        def save_best(current_model, metrics) -> None:
            save_checkpoint(
                args.checkpoint,
                current_model,
                source_vocabulary,
                target_vocabulary,
                source_tokenizer,
                target_tokenizer,
                metadata={
                    "best_epoch": metrics.epoch,
                    "validation_loss": metrics.validation_loss,
                    "model_config": asdict(model_config),
                    "training_config": asdict(training_config),
                    "loss_aggregation": "non-padding-token-mean",
                    "validation_teacher_forcing": "training",
                    "source_typo_probability": args.source_typo_probability,
                    "source_typo_backend": args.source_typo_backend,
                    "partition_scheme": "normalized-source-v1",
                },
            )
            print(f"  Saved improved checkpoint to {args.checkpoint}")

        latest_checkpoint = args.checkpoint.with_name(
            f"{args.checkpoint.stem}.latest{args.checkpoint.suffix}"
        )

        def save_progress(current_model, state) -> None:
            step = state["global_step"]
            save_checkpoint(
                latest_checkpoint,
                current_model,
                source_vocabulary,
                target_vocabulary,
                source_tokenizer,
                target_tokenizer,
                metadata={
                    "global_step": step,
                    "incomplete": True,
                    "model_config": asdict(model_config),
                    "training_config": asdict(training_config),
                    "loss_aggregation": "non-padding-token-mean",
                    "validation_teacher_forcing": "training",
                    "source_typo_probability": args.source_typo_probability,
                    "source_typo_backend": args.source_typo_backend,
                    "partition_scheme": "normalized-source-v1",
                },
                training_state=state,
            )
            print(f"  Saved progress checkpoint at step {step:,}")

        def print_epoch(metrics) -> None:
            print(
                f"Interval {metrics.epoch:03d}/{training_config.epochs}: "
                f"train={metrics.train_loss:.4f}, "
                f"validation={metrics.validation_loss:.4f}, "
                f"teacher_forcing={metrics.teacher_forcing_ratio:.3f}"
            )

        def print_progress(epoch, step, average_loss) -> None:
            print(
                f"Interval {epoch:03d} step {step:,}: "
                f"train={average_loss:.4f}"
            )

        history = train_streaming_model(
            model,
            training_data,
            validation_data,
            training_config,
            num_workers=args.data_workers,
            validation_steps=args.validation_steps,
            steps_per_epoch=args.steps_per_epoch or None,
            checkpoint_interval_steps=args.checkpoint_interval,
            log_interval_steps=args.log_interval,
            on_training_checkpoint=save_progress,
            resume_state=restored.training_state if restored else None,
            reset_patience=args.reset_patience,
            resume_tf_decay_epochs=args.resume_tf_decay_epochs,
            on_improvement=save_best,
            on_epoch=print_epoch,
            on_log=print_progress,
        )
        if restored and history:
            # The trainer returns the best weights, including a best model from
            # before this invocation when resuming to a different output path.
            best = history[0]
            for metrics in history[1:]:
                if metrics.validation_loss < best.validation_loss - training_config.min_delta:
                    best = metrics
            save_best(model, best)
        metrics_path = args.metrics_output or args.checkpoint.with_suffix(".metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps([asdict(metrics) for metrics in history], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Training finished in {format_duration(time.monotonic() - started_at)}")
        print(f"Metrics written to {metrics_path}")

        if args.evaluation_size:
            translator = Translator.from_checkpoint(args.checkpoint, device)
            test_data = IndexedTranslationDataset(**dataset_arguments, partition="test")
            indices = random.Random(args.seed).sample(range(len(test_data)),
                                                      min(args.evaluation_size, len(test_data)))
            test_pairs = [test_data.raw_pair(index) for index in indices]
            if not test_pairs:
                raise ValueError("the test partition produced no examples")
            hypotheses = [translator.translate(source) for source, _ in test_pairs]
            references = [target for _, target in test_pairs]
            scores = translation_scores(hypotheses, references)
            print(
                f"Held-out evaluation ({len(test_pairs)} examples): "
                f"BLEU={scores['bleu']:.2f}, chrF={scores['chrf']:.2f}"
            )
        return 0
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def _require_files(*paths: Path) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required file not found: {path}")


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
