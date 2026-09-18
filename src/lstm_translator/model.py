"""Bidirectional LSTM encoder-decoder used by the translation notebooks."""

from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset

from .typo import TypoGenerator

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

    def to_dict(self) -> dict:
        return {"tokens": [self.index_to_token[index] for index in range(len(self))]}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        tokens = data["tokens"]
        if tokens[: len(cls.SPECIAL_TOKENS)] != list(cls.SPECIAL_TOKENS):
            raise ValueError("checkpoint vocabulary has incompatible special tokens")
        return cls(tokens[len(cls.SPECIAL_TOKENS) :])


class TranslationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Turn aligned sequences into tensors, optionally corrupting raw sources."""

    def __init__(
        self,
        sources: Sequence[Sequence[str]],
        targets: Sequence[Sequence[str]],
        source_vocabulary: Vocabulary,
        target_vocabulary: Vocabulary,
        *,
        source_texts: Sequence[str] | None = None,
        source_tokenizer=None,
        source_typo_generator: TypoGenerator | None = None,
    ) -> None:
        if len(sources) != len(targets):
            raise ValueError("sources and targets must contain the same number of rows")
        self.sources = sources
        self.targets = targets
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary
        if source_texts is not None and len(source_texts) != len(sources):
            raise ValueError("source_texts must contain one raw source per row")
        if source_typo_generator is not None and source_texts is None:
            raise ValueError("source_texts are required for source typo augmentation")
        if source_texts is not None and source_tokenizer is None:
            raise ValueError("source_tokenizer is required for raw source texts")
        self.source_texts = source_texts
        self.source_tokenizer = source_tokenizer
        self.source_typo_generator = source_typo_generator

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if self.source_texts is None:
            source_tokens = self.sources[index]
        else:
            source_text = self.source_texts[index]
            if self.source_typo_generator is not None:
                source_text = self.source_typo_generator(source_text)
            source_tokens = self.source_tokenizer.encode(source_text)
        source = self.source_vocabulary.encode(source_tokens) + [EOS_INDEX]
        target = [SOS_INDEX, *self.target_vocabulary.encode(self.targets[index]), EOS_INDEX]
        return torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def collate_translation_batch(
    batch: Sequence[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a batch and retain the true source lengths for packed encoding."""
    sources, targets = zip(*batch)
    source_lengths = torch.tensor([source.numel() for source in sources], dtype=torch.long)
    padded_sources = nn.utils.rnn.pad_sequence(
        sources, batch_first=True, padding_value=PAD_INDEX
    )
    padded_targets = nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=PAD_INDEX
    )
    return padded_sources, padded_targets, source_lengths


@dataclass(frozen=True)
class Seq2SeqConfig:
    hidden_size: int = 128
    num_layers: int = 4
    embedding_dim: int = 42
    dropout: float = 0.0
    attention: bool = True


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

    def forward(
        self,
        inputs: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if lengths.numel() != inputs.size(0):
            raise ValueError("one source length is required per batch item")
        lengths = lengths.detach().cpu()
        if bool((lengths < 1).any()) or bool((lengths > inputs.size(1)).any()):
            raise ValueError("source lengths must be within the padded sequence width")
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        # Attention needs the per-token encoder outputs, while packing keeps
        # padding from corrupting the final recurrent state.
        packed_outputs, hidden = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=inputs.size(1),
        )
        return outputs, hidden


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention over bidirectional encoder outputs."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.encoder_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.decoder_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: Tensor, encoder_outputs: Tensor, mask: Tensor,
                projected_encoder: Tensor | None = None) -> Tensor:
        if projected_encoder is None:
            projected_encoder = self.encoder_projection(encoder_outputs)
        scores = self.energy(
            torch.tanh(
                projected_encoder
                + self.decoder_projection(query).unsqueeze(1)
            )
        ).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), encoder_outputs)


class Decoder(nn.Module):
    def __init__(self, vocabulary_size: int, config: Seq2SeqConfig) -> None:
        super().__init__()
        decoder_hidden_size = config.hidden_size * 2
        self.attention = AdditiveAttention(decoder_hidden_size) if config.attention else None
        self.embedding = nn.Embedding(vocabulary_size, config.embedding_dim, padding_idx=PAD_INDEX)
        self.lstm = nn.LSTM(
            config.embedding_dim + (decoder_hidden_size if config.attention else 0),
            decoder_hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        output_size = decoder_hidden_size * 2 if config.attention else decoder_hidden_size
        self.output = nn.Linear(output_size, vocabulary_size)

    def forward(
        self,
        inputs: Tensor,
        hidden: tuple[Tensor, Tensor],
        encoder_outputs: Tensor,
        source_mask: Tensor,
        projected_encoder: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        embedded = self.embedding(inputs)
        if self.attention is not None:
            query = hidden[0][-1]
            context = self.attention(query, encoder_outputs, source_mask, projected_encoder)
            recurrent_input = torch.cat((embedded, context), dim=-1)
        else:
            context = None
            recurrent_input = embedded

        outputs, hidden = self.lstm(recurrent_input, hidden)
        if context is not None:
            outputs = torch.cat((outputs, context), dim=-1)
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

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_lengths: Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tensor:
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")
        batch_size, target_length = target.shape
        outputs: list[Tensor] = []
        if source_lengths is None:
            source_lengths = source.ne(PAD_INDEX).sum(dim=1)
        source_mask = source.ne(PAD_INDEX)
        encoder_outputs, encoder_hidden = self.encoder(source, source_lengths)
        hidden = self._bridge_hidden(encoder_hidden)
        # Shared across decoder steps, with autograd accumulating every use.
        projected_encoder = (
            self.decoder.attention.encoder_projection(encoder_outputs)
            if self.decoder.attention is not None else None
        )

        decoder_input = target[:, :1]
        for step in range(target_length - 1):
            predictions, hidden = self.decoder(
                decoder_input,
                hidden,
                encoder_outputs,
                source_mask,
                projected_encoder,
            )
            # Stack once instead of building a chain of full-output slice writes.
            outputs.append(predictions[:, 0])
            if teacher_forcing_ratio == 1.0:
                decoder_input = target[:, step + 1 : step + 2]
            else:
                predicted = predictions.argmax(-1)
                if teacher_forcing_ratio == 0.0:
                    decoder_input = predicted
                else:
                    teacher_mask = torch.rand(batch_size, 1, device=source.device) < teacher_forcing_ratio
                    decoder_input = torch.where(
                        teacher_mask,
                        target[:, step + 1 : step + 2],
                        predicted,
                    )
        if not outputs:
            return self.decoder.output.weight.new_zeros(batch_size, 0, self.target_vocabulary_size)
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def generate(
        self,
        source: Tensor,
        source_lengths: Tensor | None = None,
        max_length: int = 100,
    ) -> Tensor:
        """Greedily generate target indices, stopping after ``<EOS>``."""
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if source_lengths is None:
            source_lengths = source.ne(PAD_INDEX).sum(dim=1)
        source_mask = source.ne(PAD_INDEX)
        encoder_outputs, encoder_hidden = self.encoder(source, source_lengths)
        hidden = self._bridge_hidden(encoder_hidden)

        projected_encoder = (
            self.decoder.attention.encoder_projection(encoder_outputs)
            if self.decoder.attention is not None else None
        )

        batch_size = source.size(0)
        decoder_input = torch.full(
            (batch_size, 1), SOS_INDEX, dtype=torch.long, device=source.device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source.device)
        generated: list[Tensor] = []

        for _ in range(max_length):
            logits, hidden = self.decoder(
                decoder_input,
                hidden,
                encoder_outputs,
                source_mask,
                projected_encoder,
            )
            next_token = logits[:, 0].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, EOS_INDEX), next_token)
            generated.append(next_token)
            finished |= next_token.eq(EOS_INDEX)
            if bool(finished.all()):
                break
            decoder_input = next_token.unsqueeze(1)

        return torch.stack(generated, dim=1)
