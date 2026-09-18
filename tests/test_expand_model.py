from pathlib import Path
import runpy

import pytest
import torch

from lstm_translator import BPETokenizer, Seq2Seq, Seq2SeqConfig, Vocabulary, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("attention,depth", [(True, 1), (False, 2), (True, 6)])
def test_expansion_preserves_weights_and_can_train(tmp_path, attention, depth):
    torch.set_num_threads(1)
    tokenizer = BPETokenizer()
    tokenizer.train(["hello"], n_merges=2)
    vocab = Vocabulary(tokenizer.tokens)
    model = Seq2Seq(len(vocab), len(vocab), Seq2SeqConfig(
        hidden_size=4, embedding_dim=3, num_layers=depth, attention=attention))
    source = tmp_path / "old.pth"
    output = tmp_path / "expanded.pth"
    save_checkpoint(source, model, vocab, vocab, tokenizer, tokenizer,
                    metadata={"epoch": 20}, training_state={"epoch": 21})
    original_bytes = source.read_bytes()
    main = runpy.run_path(str(ROOT / "scripts/expand_model.py"))["main"]
    assert main(["--checkpoint", str(source), "--output", str(output),
                 "--add-layers", "2"]) == 0
    bundle = load_checkpoint(output)
    assert bundle.model.config.num_layers == depth + 2
    assert bundle.model.config.hidden_size == 4
    assert bundle.training_state is None
    assert bundle.source_vocabulary.to_dict() == vocab.to_dict()
    assert bundle.target_tokenizer.to_dict() == tokenizer.to_dict()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, bundle.model.state_dict()[name]), name
    duplicate = tmp_path / "duplicate.pth"
    assert main(["--checkpoint", str(source), "--output", str(duplicate),
                 "--layers", str(depth + 2)]) == 0
    duplicate_state = load_checkpoint(duplicate).model.state_dict()
    for name, tensor in bundle.model.state_dict().items():
        assert torch.equal(tensor, duplicate_state[name])
    bundle.model.train()
    logits = bundle.model(torch.tensor([[4, 2]]), torch.tensor([[1, 4, 2]]))
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bundle.model.parameters())
    assert main(["--checkpoint", str(source), "--output", str(output), "--add-layers", "2"]) == 1
    assert main(["--checkpoint", str(source), "--output", str(tmp_path / "bad.pth"),
                 "--layers", str(depth)]) == 1
    assert source.read_bytes() == original_bytes


def test_train_from_expanded_checkpoint(tmp_path):
    torch.set_num_threads(1)
    tokenizer = BPETokenizer()
    tokenizer.train(["hello"], n_merges=2)
    vocab = Vocabulary(tokenizer.tokens)
    source = tmp_path / "old.pth"
    save_checkpoint(source, Seq2Seq(len(vocab), len(vocab), Seq2SeqConfig(
        hidden_size=4, embedding_dim=3, num_layers=1)), vocab, vocab, tokenizer, tokenizer)
    expanded = tmp_path / "expanded.pth"
    expand = runpy.run_path(str(ROOT / "scripts/expand_model.py"))["main"]
    assert expand(["--checkpoint", str(source), "--output", str(expanded), "--add-layers", "2"]) == 0
    corpus = tmp_path / "data.tsv"
    corpus.write_text("".join(f"hello {i}\thello {i}\n" for i in range(100)))
    train = runpy.run_path(str(ROOT / "scripts/train.py"))["main"]
    assert train(["--data", str(corpus), "--init-checkpoint", str(expanded),
                  "--checkpoint", str(tmp_path / "trained.pth"), "--device", "cpu",
                  "--steps-per-epoch", "1", "--validation-steps", "1", "--epochs", "1",
                  "--batch-size", "2", "--evaluation-size", "0",
                  "--validation-fraction", "0.2"]) == 0
    trained = load_checkpoint(tmp_path / "trained.latest.pth")
    assert trained.model.config == load_checkpoint(expanded).model.config
    assert trained.training_state["global_step"] == 1
    assert trained.training_state["optimizer"]["state"]
