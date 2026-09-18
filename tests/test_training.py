from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader

from lstm_translator import Seq2Seq, Seq2SeqConfig, TrainingConfig, Vocabulary, train_model
from lstm_translator.model import collate_translation_batch
from lstm_translator.training import _validate_config, _validation_loss, _streaming_validation_loss


def test_training_progress_is_opt_in():
    assert TrainingConfig().show_progress is False


def test_training_progress_can_be_enabled():
    assert TrainingConfig(show_progress=True).show_progress is True


@pytest.mark.parametrize("name,value", [
    ("learning_rate", float("nan")), ("learning_rate", float("inf")),
    ("gradient_clip", float("nan")), ("min_delta", float("inf")),
    ("min_delta", -1), ("teacher_forcing_decay_epochs", 0),
    ("teacher_forcing_decay_epochs", -1), ("epochs", 1.5),
])
def test_invalid_training_settings(name, value):
    with pytest.raises(ValueError, match=name):
        _validate_config(replace(TrainingConfig(), **{name: value}))


def test_validation_is_token_weighted_and_batch_invariant():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model = Seq2Seq(8, 8, Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3)).eval()
    pairs = [(torch.tensor([4, 2]), torch.tensor([1, 4, 2])),
             (torch.tensor([5, 2]), torch.tensor([1, 5, 5, 5, 5, 5, 2])),
             (torch.tensor([6, 2]), torch.tensor([1, 6, 2]))]
    total = tokens = 0
    with torch.no_grad():
        for source, target in pairs:
            output = model(source[None], target[None], teacher_forcing_ratio=0.0)
            total += torch.nn.functional.cross_entropy(output.reshape(-1, 8), target[1:], reduction="sum").item()
            tokens += len(target) - 1
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    for size in [1, 2, 3]:
        loader = DataLoader(pairs, batch_size=size, collate_fn=collate_translation_batch)
        for validation in [lambda: _validation_loss(model, loader, criterion, "cpu"),
                           lambda: _streaming_validation_loss(model, loader, criterion, "cpu", None)]:
            assert validation() == pytest.approx(total / tokens, abs=1e-6)


def test_training_loss_weights_tokens_and_groups_sources(monkeypatch):
    torch.set_num_threads(1)
    vocab = Vocabulary(["a", "b", "c"])
    model = Seq2Seq(len(vocab), len(vocab), Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3))
    sources = [["a"], ["a"], ["b"], ["c"]]
    targets = [["a"], ["b", "b", "b", "b"], ["b"], ["c", "c"]]
    losses, counts, seen = [], [], []
    original = torch.nn.CrossEntropyLoss.forward
    def capture(criterion, predictions, target):
        loss = original(criterion, predictions, target)
        if model.training:
            losses.append(loss.item()); counts.append(target.ne(0).sum().item())
        return loss
    monkeypatch.setattr(torch.nn.CrossEntropyLoss, "forward", capture)
    def validate(current, loader, criterion, device):
        training_sources = set(seen)
        held_out = {tuple(source.tolist()) for source, _ in loader.dataset}
        assert training_sources.isdisjoint(held_out)
        return 1.0
    monkeypatch.setattr("lstm_translator.training._validation_loss", validate)
    hook = model.register_forward_pre_hook(lambda module, args: seen.extend(tuple(row.tolist()) for row in args[0]))
    try:
        history = train_model(model, sources, targets, vocab, vocab,
                              TrainingConfig(epochs=1, batch_size=1, seed=7))
    finally:
        hook.remove()
    assert history[0].train_loss == pytest.approx(sum(v * n for v, n in zip(losses, counts)) / sum(counts))


def test_nonfinite_loss_stops_before_optimizer_update(monkeypatch):
    torch.set_num_threads(1)
    model = Seq2Seq(6, 6, Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3))
    with torch.no_grad():
        model.decoder.output.bias.fill_(float("nan"))
    updates = []
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args, **kwargs: updates.append(True))
    vocab = Vocabulary(["a", "b"])
    with pytest.raises(ValueError, match="non-finite loss"):
        train_model(model, [["a"], ["b"]], [["a"], ["b"]], vocab, vocab,
                    TrainingConfig(epochs=1))
    assert not updates


def test_raw_source_groups_precede_lossy_tokenization(monkeypatch):
    torch.set_num_threads(1)
    vocab = Vocabulary(["a", "b", "c", "d"])
    model = Seq2Seq(len(vocab), len(vocab), Seq2SeqConfig(hidden_size=2, num_layers=1, embedding_dim=2))
    sources = [["a"], ["b"], ["c"], ["d"]]
    groups = ["café", "CAFE\u0301", "other", "goodbye"]
    captured = []
    def validate(current, loader, criterion, device):
        captured.extend(loader.dataset.indices)
        return 1.0
    monkeypatch.setattr("lstm_translator.training._validation_loss", validate)
    train_model(model, sources, sources, vocab, vocab,
                TrainingConfig(epochs=1, batch_size=1, seed=5), source_groups=groups)
    assert (0 in captured) == (1 in captured)
    with pytest.raises(ValueError, match="source_groups"):
        train_model(model, sources, sources, vocab, vocab, source_groups=["missing rows"])
