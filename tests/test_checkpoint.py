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
