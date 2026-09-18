from lstm_translator import BPETokenizer


def test_tokenizer_round_trip(tmp_path):
    tokenizer = BPETokenizer()
    tokenizer.train(["hello hello world"], n_merges=5)
    destination = tmp_path / "tokenizer.json"
    tokenizer.save(destination)

    restored = BPETokenizer.load(destination)

    assert restored.encode("hello world") == tokenizer.encode("hello world")


def test_batch_matches_individual_encoding():
    tokenizer = BPETokenizer()
    tokenizer.train(["one two three"], n_merges=3)
    texts = ["one two", "three"]

    assert tokenizer.encode_batch(texts) == [tokenizer.encode(text) for text in texts]


def test_punctuation_is_not_silently_discarded():
    tokenizer = BPETokenizer()
    tokenizer.train(["hello!"], n_merges=0)
    encoded = tokenizer.encode("hello!")
    assert any("!" in token for token in encoded)
    assert tokenizer.decode(encoded) == "hello!"


def test_cached_result_cannot_be_mutated():
    tokenizer = BPETokenizer()
    tokenizer.train(["hello"], n_merges=2)
    first = tokenizer.encode_word("hello")
    first.clear()
    assert tokenizer.encode_word("hello")


def test_batch_larger_than_cache_limit():
    tokenizer = BPETokenizer(max_cache_size=1)
    tokenizer.train(["one two three"], n_merges=0)
    assert len(tokenizer.encode_batch(["one two three"])[0]) > 0


def test_unicode_and_underscores_survive_roundtrip(tmp_path):
    tokenizer = BPETokenizer()
    tokenizer.train(["foo_bar café"], n_merges=10)
    text = "foo_bar cafe\u0301"
    assert tokenizer.decode(tokenizer.encode(text)) == "foo_bar café"
    assert tokenizer.encode(text) == tokenizer.encode("foo_bar café")
    assert tokenizer.encode_batch([text]) == [tokenizer.encode(text)]
    tokenizer.save(tmp_path / "tokens.json")
    assert BPETokenizer.load(tmp_path / "tokens.json").encode(text) == tokenizer.encode(text)


def test_legacy_tokenizer_encoding_is_preserved():
    state = {"format_version": 2, "rules": [], "tokens": list("foobar"), "ukn_token": "?"}
    tokenizer = BPETokenizer.from_dict(state)
    assert tokenizer.decode(tokenizer.encode("foo_bar")) == "foo bar"
    assert tokenizer.ukn_token == "?"
    assert BPETokenizer.from_dict(tokenizer.to_dict()).encode("foo_bar") == tokenizer.encode("foo_bar")


def test_unsupported_tokenizer_versions_fail():
    import pytest
    for version in [0, 4, 999, True, "2"]:
        with pytest.raises(ValueError, match="format version"):
            BPETokenizer.from_dict({"format_version": version, "rules": [], "tokens": []})


def test_training_is_independent_of_hash_seed():
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    tokenizer_path = Path(__file__).resolve().parents[1] / "src/lstm_translator/tokenizer.py"
    code = '''
import contextlib, importlib.util, io, json, sys
spec = importlib.util.spec_from_file_location("tokenizer", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tokenizer = module.BPETokenizer()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    tokenizer.train(["abc abd abe abf abg ach adh aeh afh agh"], n_merges=15)
print(json.dumps(tokenizer.to_dict()))
'''
    states = [json.loads(subprocess.check_output(
        [sys.executable, "-c", code, str(tokenizer_path)],
        env={**os.environ, "PYTHONHASHSEED": seed}, text=True,
    )) for seed in ("1", "2", "3")]
    assert states[0] == states[1] == states[2]


def test_parallel_batch_matches_sequential_with_bounded_cache():
    texts = ["foo_bar café", "cafe\u0301 foo_bar", "hello world", "foo_bar", ""]
    for cache_size in (0, 1, 100):
        tokenizer = BPETokenizer(max_cache_size=cache_size)
        tokenizer.train(texts, n_merges=10)
        expected = [tokenizer.encode(text) for text in texts]
        # Exercise both cached words and new words in the parallel path.
        tokenizer._encode_cache.clear()
        tokenizer.encode("foo_bar")
        assert tokenizer.encode_batch(texts, num_workers=2) == expected
        assert len(tokenizer._encode_cache) <= cache_size
