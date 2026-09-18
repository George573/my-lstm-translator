import pytest
import torch

from lstm_translator import (
    BPETokenizer,
    Seq2Seq,
    Seq2SeqConfig,
    Vocabulary,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_restores_complete_inference_state(tmp_path):
    source_tokenizer = BPETokenizer()
    target_tokenizer = BPETokenizer()
    source_tokenizer.train(["hello"], n_merges=2)
    target_tokenizer.train(["bonjour"], n_merges=2)
    source_vocabulary = Vocabulary(source_tokenizer.tokens)
    target_vocabulary = Vocabulary(target_tokenizer.tokens)
    model = Seq2Seq(
        len(source_vocabulary),
        len(target_vocabulary),
        Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3),
    )
    path = tmp_path / "model.pth"

    save_checkpoint(
        path,
        model,
        source_vocabulary,
        target_vocabulary,
        source_tokenizer,
        target_tokenizer,
        metadata={"epoch": 1},
    )
    restored = load_checkpoint(path)

    assert restored.source_vocabulary.to_dict() == source_vocabulary.to_dict()
    assert restored.target_tokenizer.encode("bonjour") == target_tokenizer.encode("bonjour")
    assert restored.model.config == model.config
    assert restored.metadata == {"epoch": 1}


def test_legacy_weights_only_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "legacy.pth"
    torch.save({"some.weight": torch.ones(1)}, path)

    with pytest.raises(ValueError, match="legacy weights-only"):
        load_checkpoint(path)


def test_checkpoint_does_not_retry_unrestricted_loading(monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise TypeError("invalid payload")
    monkeypatch.setattr(torch, "load", fail)
    with pytest.raises(TypeError, match="invalid payload"):
        load_checkpoint("unused.pth")
    assert len(calls) == 1
    assert calls[0]["weights_only"] is True


def test_translator_loads_on_cpu_and_discards_training_state(monkeypatch):
    from lstm_translator import LoadedCheckpoint, Translator
    tokenizer = BPETokenizer()
    vocabulary = Vocabulary()
    model = Seq2Seq(4, 4, Seq2SeqConfig(hidden_size=2, num_layers=1, embedding_dim=2))
    bundle = LoadedCheckpoint(model, vocabulary, vocabulary, tokenizer, tokenizer, {}, {"optimizer": "large"})
    loads, moves = [], []
    def load(path, device):
        loads.append(device)
        return bundle
    def move(device):
        assert bundle.training_state is None
        moves.append(str(device))
        return model
    monkeypatch.setattr("lstm_translator.inference.load_checkpoint", load)
    monkeypatch.setattr(model, "to", move)
    Translator.from_checkpoint("unused.pth", "cuda")
    assert loads == ["cpu"]
    assert moves == ["cuda"]
