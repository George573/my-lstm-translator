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
