from pathlib import Path
import runpy

import torch

from lstm_translator import BPETokenizer, load_checkpoint


def test_cli_checkpoint_roundtrip_continues_coverage(tmp_path):
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
