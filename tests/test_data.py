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
