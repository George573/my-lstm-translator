"""Reproducible training and validation utilities."""

from dataclasses import asdict, dataclass
from itertools import islice
import random
from typing import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
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
) -> list[EpochMetrics]:
    """Train with a deterministic split, full epoch coverage, and early stopping."""
    config = config or TrainingConfig()
    _validate_config(config)
    seed_everything(config.seed)
    dataset = TranslationDataset(
        sources, targets, source_vocabulary, target_vocabulary
    )
    if len(dataset) < 2:
        raise ValueError("training requires at least two aligned examples")

    validation_size = max(1, int(len(dataset) * config.validation_fraction))
    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(config.seed)
    training_data, validation_data = random_split(
        dataset, [training_size, validation_size], generator=generator
    )
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
        progress = tqdm(
            training_loader,
            desc=f"Epoch {epoch}/{config.epochs}",
            unit="batch",
            disable=not config.show_progress,
        )
        for source, target, lengths in progress:
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(source, target, lengths, teacher_forcing_ratio=ratio)
            loss = criterion(
                predictions.reshape(-1, model.target_vocabulary_size),
                target[:, 1:].reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            training_loss += loss.item()
            if config.show_progress:
                progress.set_postfix(loss=f"{loss.item():.4f}")

        validation_loss = _validation_loss(model, validation_loader, criterion, device)
        metrics = EpochMetrics(
            epoch,
            training_loss / len(training_loader),
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
    on_training_checkpoint: Callable[[Seq2Seq, dict], None] | None = None,
) -> list[EpochMetrics]:
    """Train consecutive intervals of a global permutation without replacement.

    An interval ends at its step limit or the end of a dataset pass. Validation
    does not reset coverage. Teacher forcing follows fractional dataset passes.
    """
    config = config or TrainingConfig(epochs=1)
    _validate_config(config)
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
    pending_ratio = None
    settings = {key: value for key, value in asdict(config).items()
                if key not in {"epochs", "show_progress"}}
    settings.update(steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
    fingerprints = [(training_data.fingerprint, training_data.partition),
                    (validation_data.fingerprint, validation_data.partition)]
    if resume_state is not None:
        if resume_state["settings"] != settings or resume_state["fingerprints"] != fingerprints:
            raise ValueError("resume settings or corpus do not match checkpoint")
        sampler.load_state_dict(resume_state["sampler"])
        optimizer.load_state_dict(resume_state["optimizer"])
        global_step = resume_state["global_step"]
        start_epoch = resume_state["epoch"]
        pending_steps, pending_loss = resume_state["interval_steps"], resume_state["interval_loss"]
        pending_ratio = resume_state["teacher_forcing_ratio"]
        history = [EpochMetrics(**item) for item in resume_state["history"]]
        best_state, best_loss = resume_state["best_state"], resume_state["best_loss"]
        stale_epochs = resume_state["stale_intervals"]
        random.setstate(resume_state["python_rng"])
        torch.set_rng_state(resume_state["torch_rng"].cpu())
        if torch.cuda.is_available() and resume_state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in resume_state["cuda_rng"]])
        if device.type == "mps" and resume_state.get("mps_rng") is not None:
            torch.mps.set_rng_state(resume_state["mps_rng"].cpu())

    def save_training_state(epoch, steps, loss):
        if on_training_checkpoint is not None:
            on_training_checkpoint(model, dict(
                settings=settings, fingerprints=fingerprints,
                sampler=sampler.state_dict(), optimizer=optimizer.state_dict(),
                global_step=global_step, epoch=epoch, interval_steps=steps,
                teacher_forcing_ratio=ratio,
                interval_loss=loss, history=[asdict(item) for item in history],
                best_state=best_state, best_loss=best_loss, stale_intervals=stale_epochs,
                python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                mps_rng=torch.mps.get_rng_state() if device.type == "mps" else None,
            ))

    for epoch in range(start_epoch, config.epochs + 1):
        if stale_epochs >= config.patience:
            break
        if sampler.cursor == sampler.size and pending_steps == 0:
            sampler.next_cycle()
        ratio = _teacher_forcing_ratio(sampler.cycle + sampler.cursor / sampler.size + 1, config)
        if pending_steps and pending_ratio is not None:
            ratio = pending_ratio
        model.train()
        total_loss = pending_loss
        epoch_steps = pending_steps
        pending_loss, pending_steps = 0.0, 0
        interval_loss = 0.0
        log_steps = 0
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
            ratio = _teacher_forcing_ratio(sampler.cycle + sampler.cursor / sampler.size + 1, config)
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # Packed encoder lengths stay on CPU.
            optimizer.zero_grad(set_to_none=True)
            predictions = model(source, target, lengths, teacher_forcing_ratio=ratio)
            loss = criterion(
                predictions.reshape(-1, model.target_vocabulary_size),
                target[:, 1:].reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            sampler.commit(source.size(0))

            value = loss.item()
            total_loss += value
            interval_loss += value
            log_steps += 1
            epoch_steps += 1
            global_step += 1
            if config.show_progress:
                progress.set_postfix(loss=f"{total_loss / epoch_steps:.4f}")
            if on_log is not None and global_step % log_interval_steps == 0:
                on_log(epoch, global_step, interval_loss / log_steps)
                interval_loss = 0.0
                log_steps = 0
            if (
                on_checkpoint is not None
                and checkpoint_interval_steps is not None
                and global_step % checkpoint_interval_steps == 0
            ):
                on_checkpoint(model, global_step)
            if checkpoint_interval_steps is not None and global_step % checkpoint_interval_steps == 0:
                save_training_state(epoch, epoch_steps, total_loss)
            if steps_per_epoch is not None and epoch_steps >= steps_per_epoch:
                break

        if epoch_steps == 0:
            raise ValueError("the training partition produced no examples")
        validation_loss = _streaming_validation_loss(
            model, validation_loader, criterion, device, validation_steps
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=total_loss / epoch_steps,
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
        save_training_state(epoch + 1, 0, 0.0)
        if stale_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _indexed_dataset(dataset):
    if isinstance(dataset, IndexedTranslationDataset):
        return dataset
    return IndexedTranslationDataset(
        dataset.path, dataset.source_tokenizer, dataset.target_tokenizer,
        dataset.source_vocabulary, dataset.target_vocabulary,
        partition=dataset.partition, validation_fraction=dataset.validation_fraction,
        test_fraction=dataset.test_fraction, seed=dataset.seed,
    )


def _validation_loss(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for source, target, lengths in loader:
            source, target = source.to(device), target.to(device)
            predictions = model(source, target, lengths, teacher_forcing_ratio=0.0)
            total += criterion(
                predictions.reshape(-1, model.target_vocabulary_size),
                target[:, 1:].reshape(-1),
            ).item()
    return total / len(loader)


def _streaming_validation_loss(model, loader, criterion, device, max_steps) -> float:
    model.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for source, target, lengths in loader:
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # Packed encoder lengths stay on CPU.
            predictions = model(source, target, lengths, teacher_forcing_ratio=0.0)
            total += criterion(
                predictions.reshape(-1, model.target_vocabulary_size),
                target[:, 1:].reshape(-1),
            ).item()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
    if steps == 0:
        raise ValueError("the validation partition produced no examples")
    return total / steps


def _teacher_forcing_ratio(epoch: int, config: TrainingConfig) -> float:
    progress = min(1.0, max(0.0, (epoch - 1) / max(1, config.teacher_forcing_decay_epochs)))
    return config.teacher_forcing_start + progress * (
        config.teacher_forcing_end - config.teacher_forcing_start
    )


def _validate_config(config: TrainingConfig) -> None:
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 0 < config.validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not 0 <= config.teacher_forcing_start <= 1 or not 0 <= config.teacher_forcing_end <= 1:
        raise ValueError("teacher forcing ratios must be in [0, 1]")
    if config.patience < 1 or config.gradient_clip <= 0 or config.learning_rate <= 0:
        raise ValueError("patience, gradient_clip, and learning_rate must be positive")
