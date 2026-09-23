"""High-level text translation interface."""

import torch

from .checkpoint import LoadedCheckpoint, load_checkpoint
from .model import EOS_INDEX, PAD_INDEX, SOS_INDEX


class Translator:
    def __init__(self, checkpoint: LoadedCheckpoint, device: str | torch.device = "cpu") -> None:
        self.bundle = checkpoint
        self.bundle.training_state = None
        self.device = torch.device(device)
        self.bundle.model.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str | torch.device = "cpu"
    ) -> "Translator":
        return cls(load_checkpoint(path, "cpu"), device)

    def translate(
        self, text: str, max_length: int = 100, *,
        do_sample: bool = False, temperature: float = 1.0,
    ) -> str:
        """Translate greedily, or sample tokens using the given temperature."""
        tokens = self.bundle.source_tokenizer.encode(text)
        indices = self.bundle.source_vocabulary.encode(tokens) + [EOS_INDEX]
        source = torch.tensor([indices], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(indices)], dtype=torch.long)
        generated = self.bundle.model.generate(
            source, lengths, max_length=max_length,
            do_sample=do_sample, temperature=temperature,
        )[0].tolist()
        generated = [index for index in generated if index not in {PAD_INDEX, SOS_INDEX, EOS_INDEX}]
        output_tokens = self.bundle.target_vocabulary.decode(generated)
        return self.bundle.target_tokenizer.decode(output_tokens)
