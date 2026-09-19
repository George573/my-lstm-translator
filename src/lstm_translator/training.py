"""Reproducible training and validation utilities."""

from dataclasses import asdict, dataclass
from itertools import islice
import math
import random
from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .model import (
    PAD_INDEX,
    Seq2Seq,
    TranslationDataset,
    Vocabulary,
    collate_translation_batch,
)
from .streaming import StreamingTranslationDataset
from .indexed import CoverageSampler, IndexedTranslationDataset
from .data import normalize_source, split_group_indices
from .typo import TypoGenerator
from .diagnostics import BatchDiagnostics

TRAINING_STATE_VERSION = 2


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    validation_fraction: float = 0.1
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.1
    teacher_forcing_decay_epochs: int = 25
    patience: int = 5
    min_delta: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    show_progress: bool = False


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    teacher_forcing_ratio: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_model(
    model: Seq2Seq,
    sources: Sequence[Sequence[str]],
    targets: Sequence[Sequence[str]],
    source_vocabulary: Vocabulary,
    target_vocabulary: Vocabulary,
    config: TrainingConfig | None = None,
    on_improvement: Callable[[Seq2Seq, EpochMetrics], None] | None = None,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
    *,
    source_groups: Sequence[str] | None = None,
    source_texts: Sequence[str] | None = None,
    source_tokenizer=None,
    source_typo_generator: TypoGenerator | None = None,
) -> list[EpochMetrics]:
    """Train with grouped validation, full epoch coverage, and early stopping.

    Pass raw source texts as ``source_groups`` to group Unicode/case variants
    before lossy tokenization. Otherwise grouping uses the supplied BPE text.
    """
    config = config or TrainingConfig()
    _validate_config(config)
    seed_everything(config.seed)
    dataset = TranslationDataset(
        sources,
        targets,
        source_vocabulary,
        target_vocabulary,
        source_texts=source_texts,
        source_tokenizer=source_tokenizer,
    )
    if len(dataset) < 2:
        raise ValueError("training requires at least two aligned examples")
    if source_groups is not None and len(source_groups) != len(dataset):
        raise ValueError("source_groups must contain one raw source per example")

    if source_groups is None:
        source_groups = source_texts or ["".join(tokens) for tokens in sources]
    group_keys = [normalize_source(source) for source in source_groups]
    training_indices, validation_indices = split_group_indices(
        group_keys, config.validation_fraction, config.seed,
    )
    generator = torch.Generator().manual_seed(config.seed)
    training_dataset = TranslationDataset(
        sources,
        targets,
        source_vocabulary,
        target_vocabulary,
        source_texts=source_texts,
        source_tokenizer=source_tokenizer,
        source_typo_generator=source_typo_generator,
    )
    training_data = Subset(training_dataset, training_indices)
    validation_data = Subset(dataset, validation_indices)
    training_loader = DataLoader(
        training_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_translation_batch,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config.batch_size,
        collate_fn=collate_translation_batch,
    )

    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[EpochMetrics] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        ratio = _teacher_forcing_ratio(epoch, config)
        model.train()
        training_loss = 0.0
        training_tokens = 0
        progress = tqdm(
            training_loader,
            desc=f"Epoch {epoch}/{config.epochs}",
            unit="batch",
            disable=not config.show_progress,
        )
        for source, target, lengths in progress:
            value, tokens = _train_batch(
                model, source, target, lengths, criterion, optimizer,
                device, ratio, config.gradient_clip,
            )
            training_loss += value * tokens
            training_tokens += tokens
            if config.show_progress:
                progress.set_postfix(loss=f"{training_loss / training_tokens:.4f}")

        validation_loss = _validation_loss(model, validation_loader, criterion, device, ratio)
        metrics = EpochMetrics(
            epoch,
            training_loss / training_tokens,
            validation_loss,
            ratio,
        )
        history.append(metrics)
        if on_epoch is not None:
            on_epoch(metrics)
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if on_improvement is not None:
                on_improvement(model, metrics)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_streaming_model(
    model: Seq2Seq,
    training_data: StreamingTranslationDataset | IndexedTranslationDataset,
    validation_data: StreamingTranslationDataset | IndexedTranslationDataset,
    config: TrainingConfig | None = None,
    *,
    num_workers: int = 0,
    validation_steps: int | None = 1_000,
    steps_per_epoch: int | None = None,
    checkpoint_interval_steps: int | None = 10_000,
    log_interval_steps: int = 100,
    on_checkpoint: Callable[[Seq2Seq, int], None] | None = None,
    on_improvement: Callable[[Seq2Seq, EpochMetrics], None] | None = None,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
    on_log: Callable[[int, int, float], None] | None = None,
    resume_state: dict | None = None,
    reset_patience: bool = False,
    resume_tf_decay_epochs: int | None = None,
    on_training_checkpoint: Callable[[Seq2Seq, dict], None] | None = None,
    debug_logger: Callable[[dict], None] | None = None,
    debug_batches: int = 3,
) -> list[EpochMetrics]:
    """Train consecutive intervals of a global permutation without replacement.

    An interval ends at its step limit or the end of a dataset pass. Validation
    does not reset coverage. New teacher-forcing schedules count these intervals.
    """
    config = config or TrainingConfig(epochs=1)
    if resume_state is not None and resume_state.get("training_state_version") != TRAINING_STATE_VERSION:
        raise ValueError(
            "checkpoint predates token-weighted metrics and source-grouped splits; "
            "use --init-checkpoint to start a fresh run, or the original code for exact resume"
        )
    if reset_patience and resume_state is None:
        raise ValueError("reset_patience requires a training checkpoint to resume")
    if resume_tf_decay_epochs is not None:
        if resume_state is None:
            raise ValueError("resume_tf_decay_epochs requires a training checkpoint to resume")
        if resume_tf_decay_epochs < 1:
            raise ValueError("resume_tf_decay_epochs must be positive")
    _validate_config(config)
    if debug_batches < 1:
        raise ValueError("debug_batches must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if validation_steps is not None and validation_steps < 1:
        raise ValueError("validation_steps must be positive or None")
    if steps_per_epoch is not None and steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive or None")
    if checkpoint_interval_steps is not None and checkpoint_interval_steps < 1:
        raise ValueError("checkpoint_interval_steps must be positive or None")
    if log_interval_steps < 1:
        raise ValueError("log_interval_steps must be positive")

    seed_everything(config.seed)
    training_data = _indexed_dataset(training_data)
    validation_data = _indexed_dataset(validation_data)
    sampler = CoverageSampler(len(training_data), config.seed)
    # Dedicated generators keep worker startup from changing model RNG state.
    training_loader = DataLoader(
        training_data,
        sampler=sampler,
        batch_size=config.batch_size,
        num_workers=num_workers,
        collate_fn=collate_translation_batch,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_count = len(validation_data)
    if validation_steps is not None:
        validation_count = min(validation_count, validation_steps * config.batch_size)
    validation_indices = random.Random(config.seed).sample(
        range(len(validation_data)), validation_count)
    validation_loader = DataLoader(
        Subset(validation_data, validation_indices),
        batch_size=config.batch_size,
        num_workers=num_workers,
        collate_fn=collate_translation_batch,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(config.seed),
    )
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[EpochMetrics] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale_epochs = 0
    global_step = 0
    start_epoch = 1
    pending_steps = 0
    pending_loss = 0.0
    pending_tokens = 0
    pending_ratio = None
    teacher_forcing_schedule = dict(
        start_epoch=1, start_ratio=config.teacher_forcing_start,
        end_ratio=config.teacher_forcing_end,
        decay_epochs=config.teacher_forcing_decay_epochs,
    )
    settings = {key: value for key, value in asdict(config).items()
                if key not in {"epochs", "show_progress"}}
    settings.update(steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
    settings["validation_teacher_forcing"] = "training"
    fingerprints = [(training_data.fingerprint, training_data.partition),
                    (validation_data.fingerprint, validation_data.partition)]
    if resume_state is not None:
        _check_resume_settings(
            resume_state, settings, fingerprints,
            allow_new_endpoint=resume_tf_decay_epochs is not None,
        )
        sampler.load_state_dict(resume_state["sampler"])
        optimizer.load_state_dict(resume_state["optimizer"])
        global_step = resume_state["global_step"]
        start_epoch = resume_state["epoch"]
        pending_steps, pending_loss = resume_state["interval_steps"], resume_state["interval_loss"]
        pending_tokens = resume_state["interval_tokens"]
        pending_ratio = resume_state["teacher_forcing_ratio"]
        teacher_forcing_schedule = resume_state.get("teacher_forcing_schedule")
        if resume_tf_decay_epochs is not None:
            teacher_forcing_schedule = dict(
                start_epoch=start_epoch,
                start_ratio=pending_ratio,
                end_ratio=config.teacher_forcing_end,
                decay_epochs=resume_tf_decay_epochs,
            )
        history = [EpochMetrics(**item) for item in resume_state["history"]]
        best_state, best_loss = resume_state["best_state"], resume_state["best_loss"]
        stale_epochs = 0 if reset_patience else resume_state["stale_intervals"]
        random.setstate(resume_state["python_rng"])
        torch.set_rng_state(resume_state["torch_rng"].cpu())
        if torch.cuda.is_available() and resume_state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in resume_state["cuda_rng"]])
        if device.type == "mps" and resume_state.get("mps_rng") is not None:
            torch.mps.set_rng_state(resume_state["mps_rng"].cpu())

    def current_teacher_forcing():
        dataset_pass = sampler.cycle + sampler.cursor / sampler.size
        # Preserve old checkpoints' schedules unless explicitly restarted.
        if teacher_forcing_schedule is None:
            return _teacher_forcing_ratio(dataset_pass + 1, config)
        schedule = teacher_forcing_schedule
        if "start_epoch" in schedule:
            progress = (epoch - schedule["start_epoch"]) / schedule["decay_epochs"]
        else:
            progress = (dataset_pass - schedule["start_pass"]) / schedule["decay_passes"]
        progress = min(1.0, max(0.0, progress))
        return schedule["start_ratio"] + progress * (schedule["end_ratio"] - schedule["start_ratio"])

    def save_training_state(epoch, steps, loss, tokens):
        if on_training_checkpoint is not None:
            on_training_checkpoint(model, dict(
                training_state_version=TRAINING_STATE_VERSION,
                settings=settings, fingerprints=fingerprints,
                sampler=sampler.state_dict(), optimizer=optimizer.state_dict(),
                global_step=global_step, epoch=epoch, interval_steps=steps,
                teacher_forcing_ratio=ratio,
                teacher_forcing_schedule=teacher_forcing_schedule,
                interval_loss=loss, interval_tokens=tokens,
                history=[asdict(item) for item in history],
                best_state=best_state, best_loss=best_loss, stale_intervals=stale_epochs,
                python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                mps_rng=torch.mps.get_rng_state() if device.type == "mps" else None,
            ))

    initial_global_step = global_step
    for epoch in range(start_epoch, config.epochs + 1):
        if stale_epochs >= config.patience:
            break
        if sampler.cursor == sampler.size and pending_steps == 0:
            sampler.next_cycle()
        ratio = current_teacher_forcing()
        if pending_steps and pending_ratio is not None:
            ratio = pending_ratio
        model.train()
        total_loss = pending_loss
        total_tokens = pending_tokens
        epoch_steps = pending_steps
        pending_loss, pending_steps = 0.0, 0
        pending_tokens = 0
        interval_loss = 0.0
        log_tokens = 0
        batches = training_loader
        if steps_per_epoch is not None:
            batches = islice(batches, max(0, steps_per_epoch - epoch_steps))
        progress = tqdm(
            batches,
            desc=f"Interval {epoch}/{config.epochs}",
            total=steps_per_epoch,
            initial=epoch_steps,
            unit="batch",
            disable=not config.show_progress,
        )
        for source, target, lengths in progress:
            ratio = current_teacher_forcing()
            value, tokens = _train_batch(
                model, source, target, lengths, criterion, optimizer,
                device, ratio, config.gradient_clip,
                diagnostics=(BatchDiagnostics(debug_logger, device, global_step + 1)
                             if debug_logger is not None and global_step - initial_global_step < debug_batches
                             else None),
            )
            sampler.commit(source.size(0))

            total_loss += value * tokens
            total_tokens += tokens
            interval_loss += value * tokens
            log_tokens += tokens
            epoch_steps += 1
            global_step += 1
            if config.show_progress:
                progress.set_postfix(loss=f"{total_loss / total_tokens:.4f}")
            if on_log is not None and global_step % log_interval_steps == 0:
                on_log(epoch, global_step, interval_loss / log_tokens)
                interval_loss = 0.0
                log_tokens = 0
            if checkpoint_interval_steps is not None and global_step % checkpoint_interval_steps == 0:
                if on_checkpoint is not None:
                    on_checkpoint(model, global_step)
                save_training_state(epoch, epoch_steps, total_loss, total_tokens)

        if epoch_steps == 0:
            raise ValueError("the training partition produced no examples")
        validation_loss = _streaming_validation_loss(
            model, validation_loader, criterion, device, validation_steps, ratio
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=total_loss / total_tokens,
            validation_loss=validation_loss,
            teacher_forcing_ratio=ratio,
        )
        history.append(metrics)
        if on_epoch is not None:
            on_epoch(metrics)
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if on_improvement is not None:
                on_improvement(model, metrics)
        else:
            stale_epochs += 1
        save_training_state(epoch + 1, 0, 0.0, 0)
        if stale_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _check_resume_settings(resume_state, settings, fingerprints, allow_new_endpoint):
    """Explain incompatible settings before restoring any training state."""
    mismatches = []
    saved_settings = resume_state["settings"]
    for key in sorted(saved_settings.keys() | settings.keys()):
        if key == "teacher_forcing_end" and allow_new_endpoint:
            continue
        if key not in saved_settings or key not in settings or saved_settings[key] != settings[key]:
            mismatches.append(
                f"{key}: checkpoint={saved_settings.get(key)!r}, current={settings.get(key)!r}"
                + (" (use --resume-tf-decay-epochs to restart decay toward a new endpoint)"
                   if key == "teacher_forcing_end" else "")
            )
    if resume_state["fingerprints"] != fingerprints:
        mismatches.append(
            "corpus fingerprint differs (includes absolute path, file size, "
            "mtime/ctime timestamps, split fractions, and seed; copying or "
            "regenerating the CSV can change it even if its contents are identical)"
        )
    if mismatches:
        raise ValueError("resume settings or corpus do not match checkpoint:\n  "
                         + "\n  ".join(mismatches))


def _indexed_dataset(dataset):
    if isinstance(dataset, IndexedTranslationDataset):
        return dataset
    return IndexedTranslationDataset(
        dataset.path, dataset.source_tokenizer, dataset.target_tokenizer,
        dataset.source_vocabulary, dataset.target_vocabulary,
        partition=dataset.partition, validation_fraction=dataset.validation_fraction,
        test_fraction=dataset.test_fraction, seed=dataset.seed,
        source_typo_generator=getattr(dataset, "source_typo_generator", None),
    )


def _train_batch(model, source, target, lengths, criterion, optimizer,
                 device, teacher_forcing_ratio, gradient_clip, *, diagnostics=None) -> tuple[float, int]:
    """Update weights once and return mean token loss plus valid-token count."""
    from contextlib import nullcontext

    with diagnostics.watch(model, source, target, lengths) if diagnostics else nullcontext():
        def phase(name):
            if diagnostics:
                diagnostics.phase(name)

        phase("transfer")
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        phase("forward")
        predictions = model(source, target, lengths, teacher_forcing_ratio=teacher_forcing_ratio)
        phase("loss")
        loss = criterion(
            predictions.reshape(-1, model.target_vocabulary_size),
            target[:, 1:].reshape(-1),
        )
        value, tokens = _loss_statistics(loss, target)
        phase("backward")
        loss.backward()
        phase("gradient_clip")
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        phase("optimizer")
        optimizer.step()
        phase("complete")
        return value, tokens


def _validation_loss(model, loader, criterion, device, teacher_forcing_ratio=0.0) -> float:
    return _streaming_validation_loss(model, loader, criterion, device, None, teacher_forcing_ratio)


def _loss_statistics(loss, target) -> tuple[float, int]:
    value = loss.item()
    if not math.isfinite(value):
        raise ValueError("non-finite loss; stopping before updating model weights")
    tokens = target[:, 1:].ne(PAD_INDEX).sum().item()
    if tokens == 0:
        raise ValueError("batch contains no valid target tokens")
    return value, tokens


def _streaming_validation_loss(model, loader, criterion, device, max_steps,
                               teacher_forcing_ratio=0.0) -> float:
    model.eval()
    total = 0.0
    total_tokens = 0
    steps = 0
    with torch.no_grad():
        for source, target, lengths in loader:
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # Packed encoder lengths stay on CPU.
            predictions = model(source, target, lengths, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(
                predictions.reshape(-1, model.target_vocabulary_size),
                target[:, 1:].reshape(-1),
            )
            value, tokens = _loss_statistics(loss, target)
            total += value * tokens
            total_tokens += tokens
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
    if steps == 0:
        raise ValueError("the validation partition produced no examples")
    return total / total_tokens


def _teacher_forcing_ratio(epoch: int, config: TrainingConfig) -> float:
    progress = min(1.0, max(0.0, (epoch - 1) / max(1, config.teacher_forcing_decay_epochs)))
    return config.teacher_forcing_start + progress * (
        config.teacher_forcing_end - config.teacher_forcing_start
    )


def _validate_config(config: TrainingConfig) -> None:
    for name in ("epochs", "batch_size", "patience", "teacher_forcing_decay_epochs"):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("learning_rate", "gradient_clip", "min_delta"):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0 or (name != "min_delta" and value == 0):
            raise ValueError(f"{name} must be finite and {'non-negative' if name == 'min_delta' else 'positive'}")
    if not 0 < config.validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not 0 <= config.teacher_forcing_start <= 1 or not 0 <= config.teacher_forcing_end <= 1:
        raise ValueError("teacher forcing ratios must be in [0, 1]")
