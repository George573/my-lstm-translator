"""Self-contained and reproducible model checkpoints."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .model import Seq2Seq, Seq2SeqConfig, Vocabulary
from .tokenizer import BPETokenizer

CHECKPOINT_VERSION = 1


@dataclass
class LoadedCheckpoint:
    model: Seq2Seq
    source_vocabulary: Vocabulary
    target_vocabulary: Vocabulary
    source_tokenizer: BPETokenizer
    target_tokenizer: BPETokenizer
    metadata: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    model: Seq2Seq,
    source_vocabulary: Vocabulary,
    target_vocabulary: Vocabulary,
    source_tokenizer: BPETokenizer,
    target_tokenizer: BPETokenizer,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save weights together with every mapping required for inference."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "source_vocabulary": source_vocabulary.to_dict(),
            "target_vocabulary": target_vocabulary.to_dict(),
            "source_tokenizer": source_tokenizer.to_dict(),
            "target_tokenizer": target_tokenizer.to_dict(),
            "metadata": dict(metadata or {}),
        },
        destination,
    )


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Load a v1 checkpoint without relying on Python pickle classes."""
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            "This is a legacy weights-only checkpoint. Retrain or migrate it with "
            "the original vocabulary before using the current inference pipeline."
        )

    source_vocabulary = Vocabulary.from_dict(payload["source_vocabulary"])
    target_vocabulary = Vocabulary.from_dict(payload["target_vocabulary"])
    model = Seq2Seq(
        len(source_vocabulary),
        len(target_vocabulary),
        Seq2SeqConfig(**payload["model_config"]),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return LoadedCheckpoint(
        model=model,
        source_vocabulary=source_vocabulary,
        target_vocabulary=target_vocabulary,
        source_tokenizer=BPETokenizer.from_dict(payload["source_tokenizer"]),
        target_tokenizer=BPETokenizer.from_dict(payload["target_tokenizer"]),
        metadata=dict(payload.get("metadata", {})),
    )
