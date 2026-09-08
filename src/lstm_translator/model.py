"""Bidirectional LSTM encoder-decoder used by the translation notebooks."""

from dataclasses import dataclass
import random
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3


class Vocabulary:
    """A deterministic mapping between BPE tokens and integer indices."""

    SPECIAL_TOKENS = ("<PAD>", "<SOS>", "<EOS>", "<UNK>")

    def __init__(self, tokens: Iterable[str] = ()) -> None:
        ordered_tokens = [*self.SPECIAL_TOKENS]
        ordered_tokens.extend(sorted(set(tokens) - set(self.SPECIAL_TOKENS)))
        self.token_to_index = {token: index for index, token in enumerate(ordered_tokens)}
        self.index_to_token = dict(enumerate(ordered_tokens))

    def __len__(self) -> int:
        return len(self.token_to_index)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_to_index.get(token, UNK_INDEX) for token in tokens]

    def decode(self, indices: Iterable[int]) -> list[str]:
        return [self.index_to_token.get(index, "<UNK>") for index in indices]


class TranslationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Turn aligned BPE sequences into source and decoder-target tensors."""

    def __init__(
        self,
        sources: Sequence[Sequence[str]],
        targets: Sequence[Sequence[str]],
        source_vocabulary: Vocabulary,
        target_vocabulary: Vocabulary,
    ) -> None:
        if len(sources) != len(targets):
            raise ValueError("sources and targets must contain the same number of rows")
        self.sources = sources
        self.targets = targets
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        source = self.source_vocabulary.encode(self.sources[index]) + [EOS_INDEX]
        target = [SOS_INDEX, *self.target_vocabulary.encode(self.targets[index]), EOS_INDEX]
        return torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)


@dataclass(frozen=True)
class Seq2SeqConfig:
    hidden_size: int = 128
    num_layers: int = 4
    embedding_dim: int = 42
    dropout: float = 0.0


class Encoder(nn.Module):
    def __init__(self, vocabulary_size: int, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config.embedding_dim, padding_idx=PAD_INDEX)
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        return self.lstm(self.embedding(inputs))


class Decoder(nn.Module):
    def __init__(self, vocabulary_size: int, config: Seq2SeqConfig) -> None:
        super().__init__()
        decoder_hidden_size = config.hidden_size * 2
        self.embedding = nn.Embedding(vocabulary_size, config.embedding_dim, padding_idx=PAD_INDEX)
        self.lstm = nn.LSTM(
            config.embedding_dim,
            decoder_hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(decoder_hidden_size, vocabulary_size)

    def forward(self, inputs: Tensor, hidden: tuple[Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        outputs, hidden = self.lstm(self.embedding(inputs), hidden)
        return self.output(outputs), hidden


class Seq2Seq(nn.Module):
    """A bidirectional encoder with an autoregressive decoder."""

    def __init__(
        self,
        source_vocabulary_size: int,
        target_vocabulary_size: int,
        config: Seq2SeqConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or Seq2SeqConfig()
        self.target_vocabulary_size = target_vocabulary_size
        self.encoder = Encoder(source_vocabulary_size, self.config)
        self.decoder = Decoder(target_vocabulary_size, self.config)

    def _bridge_hidden(self, hidden: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        def bridge(state: Tensor) -> Tensor:
            layers = self.config.num_layers
            batch_size = state.size(1)
            state = state.reshape(layers, 2, batch_size, self.config.hidden_size)
            return torch.cat((state[:, 0], state[:, 1]), dim=-1)

        return bridge(hidden[0]), bridge(hidden[1])

    def forward(self, source: Tensor, target: Tensor, teacher_forcing_ratio: float = 0.5) -> Tensor:
        batch_size, target_length = target.shape
        outputs = torch.zeros(
            batch_size,
            target_length - 1,
            self.target_vocabulary_size,
            device=source.device,
        )
        _, encoder_hidden = self.encoder(source)
        hidden = self._bridge_hidden(encoder_hidden)

        if teacher_forcing_ratio == 1.0:
            predictions, _ = self.decoder(target[:, :-1], hidden)
            return predictions

        decoder_input = target[:, :1]
        for step in range(target_length - 1):
            predictions, hidden = self.decoder(decoder_input, hidden)
            outputs[:, step] = predictions[:, 0]
            use_teacher = random.random() < teacher_forcing_ratio
            decoder_input = target[:, step + 1 : step + 2] if use_teacher else predictions.argmax(-1)
        return outputs
