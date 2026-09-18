from lstm_translator.data import (
    iter_parallel_rows,
    load_parallel_tsv,
    parallel_partition,
    split_parallel,
)


def test_load_parallel_tsv(tmp_path):
    corpus = tmp_path / "sample.tsv"
    corpus.write_text("Hello\tBonjour\tattribution\nBye\tAu revoir\tmetadata\n", encoding="utf-8")

    english, french = load_parallel_tsv(corpus, num_samples=1)

    assert english == ["Hello"]
    assert french == ["Bonjour"]


def test_zero_samples_returns_empty_lists(tmp_path):
    corpus = tmp_path / "sample.tsv"
    corpus.write_text("Hello\tBonjour\n", encoding="utf-8")
    assert load_parallel_tsv(corpus, num_samples=0) == ([], [])


def test_split_parallel_is_deterministic_and_disjoint():
    sources = [f"source-{index}" for index in range(10)]
    targets = [f"target-{index}" for index in range(10)]
    first = split_parallel(sources, targets, test_fraction=0.2, seed=7)
    second = split_parallel(sources, targets, test_fraction=0.2, seed=7)

    assert first == second
    assert set(first[0]).isdisjoint(first[2])


def test_iter_parallel_rows_supports_csv(tmp_path):
    corpus = tmp_path / "sample.csv"
    corpus.write_text('en,fr\n"Hello, world",Bonjour\nBye,Au revoir\n', encoding="utf-8")
    assert list(iter_parallel_rows(corpus)) == [
        ("Hello, world", "Bonjour"),
        ("Bye", "Au revoir"),
    ]


def test_hash_partition_is_stable():
    arguments = {
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
        "seed": 42,
    }
    assert parallel_partition("hello", "bonjour", **arguments) == parallel_partition(
        "hello", "bonjour", **arguments
    )


def test_csv_preserves_text_and_matches_streaming(tmp_path):
    from lstm_translator.data import load_parallel_csv
    corpus = tmp_path / "sample.csv"
    corpus.write_text('en,fr\nNA,oui\n001,un\n2,deux\n,absent\n', encoding="utf-8")
    sources, targets = load_parallel_csv(corpus)
    assert list(zip(sources, targets)) == list(iter_parallel_rows(corpus))
    assert sources == ["NA", "001", "2"]


def test_source_variants_stay_in_one_partition():
    from lstm_translator.data import normalize_source
    sources = [" Café  noir ", "CAFE\u0301 noir", "hello", "HELLO", "goodbye"]
    targets = ["a", "b", "c", "d", "e"]
    training, _, test, _ = split_parallel(sources, targets, test_fraction=0.5)
    assert {normalize_source(s) for s in training}.isdisjoint(normalize_source(s) for s in test)
    assert len(training) + len(test) == len(sources)
    for seed in range(20):
        options = dict(validation_fraction=0.2, test_fraction=0.3, seed=seed)
        assert parallel_partition(sources[0], "a", **options) == parallel_partition(sources[1], "b", **options)


def test_split_rejects_single_source_group():
    import pytest
    with pytest.raises(ValueError, match="distinct source"):
        split_parallel(["hello", " HELLO "], ["bonjour", "salut"])


def test_partition_rejects_nonfinite_fractions():
    import pytest
    for value in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError, match="finite"):
            parallel_partition("a", "b", validation_fraction=value, test_fraction=0.1, seed=0)
