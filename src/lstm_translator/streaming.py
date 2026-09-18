"""Constant-memory datasets for corpora containing millions of sentence pairs."""

from pathlib import Path
import random
from collections.abc import Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from .data import Partition, iter_parallel_rows, parallel_partition
from .model import EOS_INDEX, SOS_INDEX, Vocabulary
from .tokenizer import BPETokenizer
from .typo import TypoGenerator


class StreamingTranslationDataset(IterableDataset[tuple[Tensor, Tensor]]):
    """Read, partition, shuffle, and tokenize parallel examples on demand.

    Memory use is bounded by ``shuffle_buffer_size`` plus the tokenizer caches;
    it does not grow with the number of rows in the corpus.
    """

    def __init__(
        self,
        path: str | Path,
        source_tokenizer: BPETokenizer,
        target_tokenizer: BPETokenizer,
        source_vocabulary: Vocabulary,
        target_vocabulary: Vocabulary,
        *,
        partition: Partition = "train",
        validation_fraction: float = 0.01,
        test_fraction: float = 0.01,
        seed: int = 42,
        shuffle_buffer_size: int = 10_000,
        source_typo_generator: TypoGenerator | None = None,
    ) -> None:
        super().__init__()
        if shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be positive")
        if partition not in {"train", "validation", "test"}:
            raise ValueError("partition must be train, validation, or test")
        # Validate fractions once instead of once per streamed row.
        parallel_partition(
            "validation", "probe", validation_fraction=validation_fraction,
            test_fraction=test_fraction, seed=seed
        )
        self.path = Path(path)
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary
        self.partition = partition
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.source_typo_generator = source_typo_generator
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        rng = random.Random(self.seed + self.epoch * 1_000_003 + worker_id)
        examples = self._iter_worker_partition(worker_id, worker_count)

        if self.partition != "train":
            yield from examples
            return

        buffer: list[tuple[Tensor, Tensor]] = []
        for example in examples:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(example)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = example
        rng.shuffle(buffer)
        yield from buffer

    def _iter_worker_partition(
        self, worker_id: int, worker_count: int
    ) -> Iterator[tuple[Tensor, Tensor]]:
        for row_number, (source, target) in enumerate(iter_parallel_rows(self.path)):
            if row_number % worker_count != worker_id:
                continue
            if parallel_partition(
                source,
                target,
                validation_fraction=self.validation_fraction,
                test_fraction=self.test_fraction,
                seed=self.seed,
            ) != self.partition:
                continue
            if self.source_typo_generator is not None:
                source = self.source_typo_generator(source)
            source_tokens = self.source_tokenizer.encode(source)
            target_tokens = self.target_tokenizer.encode(target)
            source_indices = self.source_vocabulary.encode(source_tokens) + [EOS_INDEX]
            target_indices = [
                SOS_INDEX,
                *self.target_vocabulary.encode(target_tokens),
                EOS_INDEX,
            ]
            yield (
                torch.tensor(source_indices, dtype=torch.long),
                torch.tensor(target_indices, dtype=torch.long),
            )
