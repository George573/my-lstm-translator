"""High-level text translation interface."""

import torch

from .checkpoint import LoadedCheckpoint, load_checkpoint
from .model import EOS_INDEX, PAD_INDEX, SOS_INDEX


class Translator:
    def __init__(self, checkpoint: LoadedCheckpoint, device: str | torch.device = "cpu") -> None:
        self.bundle = checkpoint
        self.device = torch.device(device)
        self.bundle.model.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str | torch.device = "cpu"
    ) -> "Translator":
        return cls(load_checkpoint(path, device), device)

    def translate(self, text: str, max_length: int = 100) -> str:
        tokens = self.bundle.source_tokenizer.encode(text)
        indices = self.bundle.source_vocabulary.encode(tokens) + [EOS_INDEX]
        source = torch.tensor([indices], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(indices)], dtype=torch.long, device=self.device)
        generated = self.bundle.model.generate(source, lengths, max_length=max_length)[0].tolist()
        generated = [index for index in generated if index not in {PAD_INDEX, SOS_INDEX, EOS_INDEX}]
        output_tokens = self.bundle.target_vocabulary.decode(generated)
        return self.bundle.target_tokenizer.decode(output_tokens)
