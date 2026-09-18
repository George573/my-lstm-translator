import json
from pathlib import Path
import runpy

import pytest
import torch

from lstm_translator import BPETokenizer, Seq2Seq, Seq2SeqConfig, Vocabulary, save_checkpoint
from lstm_translator.evaluate_cli import main


def test_evaluate_checkpoint_and_reuse_exact_examples(tmp_path):
    torch.set_num_threads(1)
    tokenizer = BPETokenizer()
    tokenizer.train(["hello bonjour"], n_merges=0)
    vocab = Vocabulary(tokenizer.tokens)
    checkpoint = tmp_path / "model.pth"
    save_checkpoint(checkpoint, Seq2Seq(len(vocab), len(vocab), Seq2SeqConfig(
        hidden_size=2, embedding_dim=2, num_layers=1)), vocab, vocab, tokenizer, tokenizer)
    corpus = tmp_path / "pairs.tsv"
    corpus.write_text("".join(f"hello {i}\tbonjour {i}\n" for i in range(50)))
    output = tmp_path / "results.json"
    args = ["--checkpoints", str(checkpoint), "--output", str(output), "--max-length", "3", "--threads", "1"]
    assert main([*args, "--data", str(corpus), "--test-fraction", "0.5", "--sample-size", "3"]) == 0
    first = json.loads(output.read_text())
    examples = first["models"]["model.pth"]["datasets"][str(corpus)]["examples"]
    assert len(examples) == 3
    second = tmp_path / "second.json"
    assert main([*args, "--examples-from", str(output), "--output", str(second)]) == 0
    rerun = json.loads(second.read_text())
    assert rerun["partition_scheme"] == "saved-examples"
    assert rerun["models"]["model.pth"]["datasets"][str(corpus)]["examples"] == examples


def test_historical_scripts_are_import_safe(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not read reports or load checkpoints on import")
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    root = Path(__file__).resolve().parents[1]
    for script in root.glob("artifacts/evaluations/*/*.py"):
        namespace = runpy.run_path(str(script))
        assert callable(namespace["main"])


def test_significance_rejects_unaligned_examples(tmp_path):
    root = Path(__file__).resolve().parents[1]
    analyze = runpy.run_path(str(root / "artifacts/evaluations/2026-09-14-quick-tf-decay/analyze.py"))["main"]
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"models": {
        "old": {"datasets": {"test": {"examples": [{"source": "a", "reference": "b", "hypothesis": "b"}]}}},
        "new": {"datasets": {"test": {"examples": [{"source": "c", "reference": "d", "hypothesis": "d"}]}}},
    }}))
    with pytest.raises(SystemExit, match="1"):
        analyze(["--input", str(report), "--output", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()
