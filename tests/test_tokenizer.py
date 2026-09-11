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
