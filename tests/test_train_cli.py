from pathlib import Path
import runpy

import torch
import pytest

from lstm_translator import BPETokenizer, load_checkpoint


def test_cli_checkpoint_roundtrip_continues_coverage(tmp_path, capsys):
    torch.set_num_threads(1)
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train.py"))["main"]
    corpus = tmp_path / "pairs.tsv"
    corpus.write_text("".join(f"{i}\t{i}\n" for i in range(100)))
    tokenizer = BPETokenizer()
    tokenizer.train([str(i) for i in range(100)], n_merges=2)
    tokens = tmp_path / "tokens.json"
    tokenizer.save(tokens)
    output = tmp_path / "model.pth"
    args = ["--data", str(corpus), "--source-tokenizer", str(tokens),
            "--target-tokenizer", str(tokens), "--checkpoint", str(output),
            "--batch-size", "4", "--hidden-size", "4", "--layers", "1",
            "--embedding-size", "3", "--steps-per-epoch", "2",
            "--validation-steps", "1", "--validation-fraction", "0.2",
            "--test-fraction", "0.1", "--evaluation-size", "2", "--device", "cpu"]
    assert main([*args, "--epochs", "1"]) == 0
    latest = tmp_path / "model.latest.pth"
    first = load_checkpoint(latest)
    assert first.training_state["sampler"]["cursor"] == 8
    assert first.training_state["epoch"] == 2
    assert first.training_state["optimizer"]["state"]
    assert load_checkpoint(output).training_state is None
    assert main([*args, "--epochs", "2", "--resume", str(latest)]) == 0
    resumed = load_checkpoint(latest)
    assert resumed.training_state["sampler"]["cursor"] == 16
    assert resumed.training_state["sampler"]["cycle"] == 0
    assert resumed.training_state["global_step"] == 4

    assert main([*args, "--epochs", "3", "--resume", str(latest),
                 "--batch-size", "8"]) == 1
    assert "batch_size: checkpoint=4, current=8" in capsys.readouterr().err

    copied_corpus = tmp_path / "copied.tsv"
    copied_corpus.write_bytes(corpus.read_bytes())
    assert main([*args, "--epochs", "3", "--resume", str(latest),
                 "--data", str(copied_corpus)]) == 1
    error = capsys.readouterr().err
    assert "corpus fingerprint differs" in error
    assert "absolute path" in error
    assert "batch_size:" not in error


@pytest.mark.parametrize("reset", [False, True])
def test_resume_after_early_stopping(tmp_path, monkeypatch, reset):
    torch.set_num_threads(1)
    monkeypatch.setattr("lstm_translator.training._streaming_validation_loss",
                        lambda *args, **kwargs: 2.0)
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train.py"))["main"]
    corpus = tmp_path / "pairs.tsv"
    corpus.write_text("".join(f"{i}\t{i}\n" for i in range(100)))
    tokenizer = BPETokenizer()
    tokenizer.train([str(i) for i in range(100)], n_merges=2)
    tokens = tmp_path / "tokens.json"
    tokenizer.save(tokens)
    output = tmp_path / "model.pth"
    args = ["--data", str(corpus), "--source-tokenizer", str(tokens),
            "--target-tokenizer", str(tokens), "--checkpoint", str(output),
            "--batch-size", "4", "--hidden-size", "4", "--layers", "1",
            "--embedding-size", "3", "--steps-per-epoch", "2",
            "--validation-steps", "1", "--validation-fraction", "0.2",
            "--evaluation-size", "0", "--device", "cpu", "--patience", "1",
            "--epochs", "10"]
    assert main(args) == 0
    latest = tmp_path / "model.latest.pth"
    before = load_checkpoint(latest).training_state
    assert before["global_step"] == 4
    assert before["stale_intervals"] == 1
    assert main([*args, "--resume", str(latest),
                 *(["--reset-patience"] if reset else [])]) == 0
    after = load_checkpoint(latest).training_state
    assert after["global_step"] == (6 if reset else 4)
    assert after["sampler"]["cursor"] == (24 if reset else 16)
    assert after["stale_intervals"] == 1
    assert after["best_loss"] == before["best_loss"]
    for key, value in before["best_state"].items():
        assert torch.equal(after["best_state"][key], value)


@pytest.mark.parametrize("flags", [["--reset-patience"], ["--resume-tf-decay-epochs", "5"]])
def test_resume_options_require_resume(flags):
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train.py"))["main"]
    with pytest.raises(SystemExit, match="2"):
        main(flags)


@pytest.mark.parametrize("flag", ["--learning-rate", "--gradient-clip", "--min-delta"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cli_rejects_nonfinite_values(flag, value):
    parser = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train.py"))["build_parser"]()
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args([f"{flag}={value}"])
