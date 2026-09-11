from copy import deepcopy
from itertools import islice

import pytest
import torch
from torch.utils.data import DataLoader

from lstm_translator import (
    CoverageSampler, IndexedTranslationDataset, Seq2Seq, Seq2SeqConfig,
    TrainingConfig, Vocabulary, train_streaming_model,
)
from lstm_translator.data import iter_parallel_rows, parallel_partition


class Words:
    def encode(self, text):
        return text.split()


def dataset(path, partition="train", validation_fraction=0):
    vocabulary = Vocabulary([str(i) for i in range(100)])
    return IndexedTranslationDataset(
        path, Words(), Words(), vocabulary, vocabulary, partition=partition,
        validation_fraction=validation_fraction, test_fraction=0,
    )


def test_csv_record_offsets_cache_and_invalidation(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text('fr,en,extra\r\n"un\ndeux","one\ntwo",x\r\n,skip,x\r\n'
                    '"été, oui","summer, yes",x\r\n\r\n')
    first = dataset(path)
    assert [first.raw_pair(i) for i in range(len(first))] == list(iter_parallel_rows(path))
    second = dataset(path)
    assert second.offset_path == first.offset_path
    path.write_text("fr,en\nnew,nouveau\n")
    changed = dataset(path)
    assert changed.fingerprint != first.fingerprint
    assert changed.raw_pair(0) == ("nouveau", "new")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_index_matches_partition_and_filters_invalid_rows(tmp_path, newline):
    path = tmp_path / "pairs.tsv"
    text = "invalid\n\tmissing\n" + "".join(f"{i}\t{i}\n" for i in range(100))
    path.write_bytes(text.replace("\n", newline).encode())
    for partition in ("train", "validation"):
        data = dataset(path, partition, 0.2)
        expected = [pair for pair in iter_parallel_rows(path) if parallel_partition(
            *pair, validation_fraction=0.2, test_fraction=0, seed=42) == partition]
        assert [data.raw_pair(i) for i in range(len(data))] == expected


def test_sampler_prefetch_resume_and_complete_coverage():
    sampler = CoverageSampler(101, 7)
    order = list(sampler)
    assert sorted(order) == list(range(101))
    # Reading an entire pass ahead must not acknowledge any training examples.
    assert sampler.cursor == 0
    sampler.commit(13)
    restored = CoverageSampler(101, 7)
    restored.load_state_dict(sampler.state_dict())
    assert list(restored) == order[13:]
    with pytest.raises(ValueError, match="before completing"):
        restored.next_cycle()
    restored.commit(88)
    restored.next_cycle()
    assert sorted(restored) == list(range(101))
    assert list(restored) != order


@pytest.mark.parametrize("workers,context", [(0, None), (2, None), (2, "spawn")])
def test_loader_intervals_cover_each_row_once(tmp_path, workers, context):
    path = tmp_path / "pairs.tsv"
    path.write_text("".join(f"{i}\t{i}\n" for i in range(31)))
    data = dataset(path)
    data.raw_pair(0)  # Exercise handles opened before workers are forked/spawned.
    sampler = CoverageSampler(len(data))
    seen = []
    while len(sampler):
        loader = DataLoader(data, batch_size=4, sampler=sampler, num_workers=workers,
                            multiprocessing_context=context, timeout=15 if workers else 0)
        for source, target in islice(loader, 2):
            seen.extend(source[:, 0].tolist())
            sampler.commit(len(source))
    assert len(seen) == len(set(seen)) == 31


@pytest.mark.parametrize("stop_step", [1, 2, 3])
def test_training_resume_matches_uninterrupted_updates(tmp_path, stop_step):
    torch.set_num_threads(1)
    path = tmp_path / "pairs.tsv"
    path.write_text("".join(f"{i}\t{i}\n" for i in range(11)))
    data = dataset(path)
    config = TrainingConfig(epochs=4, batch_size=4, patience=20)
    torch.manual_seed(123)
    model = Seq2Seq(len(data.source_vocabulary), len(data.target_vocabulary),
                   Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3, dropout=0))
    captured = []

    def checkpoint(current, state):
        captured.append((deepcopy(current.state_dict()), deepcopy(state)))

    expected = train_streaming_model(
        model, data, data, config, steps_per_epoch=2, validation_steps=1,
        checkpoint_interval_steps=1, on_training_checkpoint=checkpoint,
    )
    final_weights = deepcopy(model.state_dict())
    weights, state = next((weights, state) for weights, state in captured
                          if state["global_step"] == stop_step and state["interval_steps"] > 0)
    model.load_state_dict(weights)
    actual = train_streaming_model(
        model, data, data, config, steps_per_epoch=2, validation_steps=1,
        checkpoint_interval_steps=1, resume_state=state,
    )
    assert actual == expected
    for name, value in model.state_dict().items():
        assert torch.equal(value, final_weights[name])
    # Completed intervals advance the same pass instead of reshuffling it.
    ends = [state["sampler"] for _, state in captured if state["interval_steps"] == 0]
    assert [(state["cycle"], state["cursor"]) for state in ends] == [(0, 8), (0, 11), (1, 8), (1, 11)]
