import csv
from pathlib import Path
import runpy

from lstm_translator import BPETokenizer


def test_filter_uses_bpe_boundary_on_both_sides_and_preserves_csv(tmp_path):
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/filter_csv.py"))["main"]
    tokenizer = BPETokenizer()
    tokenizer.train(["a", "ab"], n_merges=0)
    tokens = tmp_path / "tokens.json"
    tokenizer.save(tokens)
    assert len(tokenizer.encode("a")) == 2
    assert len(tokenizer.encode("ab")) == 3
    corpus, output = tmp_path / "input.csv", tmp_path / "output.csv"
    rows = [
        ["en", "fr", "metadata"],
        ["a\n", "a", 'comma, and "quote"'],
        ["ab", "a", "source at limit"],
        ["a", "ab", "target at limit"],
        ["a", " ", "empty"],
        ["a"],
    ]
    with corpus.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    args = ["--input", str(corpus), "--output", str(output), "--max-tokens", "3",
            "--source-tokenizer", str(tokens), "--target-tokenizer", str(tokens)]
    original = corpus.read_bytes()
    assert main(args) == 0
    with output.open(newline="") as stream:
        assert list(csv.reader(stream)) == rows[:2]
    filtered = output.read_bytes()
    assert main(args) == 1
    assert output.read_bytes() == filtered
    assert main([*args, "--output", str(corpus)]) == 1
    assert corpus.read_bytes() == original
