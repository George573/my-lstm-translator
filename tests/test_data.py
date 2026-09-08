from lstm_translator.data import load_parallel_tsv


def test_load_parallel_tsv(tmp_path):
    corpus = tmp_path / "sample.tsv"
    corpus.write_text("Hello\tBonjour\tattribution\nBye\tAu revoir\tmetadata\n", encoding="utf-8")

    english, french = load_parallel_tsv(corpus, num_samples=1)

    assert english == ["Hello"]
    assert french == ["Bonjour"]
