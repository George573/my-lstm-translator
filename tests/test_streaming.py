from lstm_translator import BPETokenizer, StreamingTranslationDataset, Vocabulary


def test_streaming_dataset_encodes_every_training_row(tmp_path):
    corpus = tmp_path / "sample.tsv"
    corpus.write_text(
        "one\tun\n"
        "two\tdeux\n"
        "three\ttrois\n",
        encoding="utf-8",
    )
    source_tokenizer = BPETokenizer()
    target_tokenizer = BPETokenizer()
    source_tokenizer.train(["one two three"], n_merges=2)
    target_tokenizer.train(["un deux trois"], n_merges=2)
    dataset = StreamingTranslationDataset(
        corpus,
        source_tokenizer,
        target_tokenizer,
        Vocabulary(source_tokenizer.tokens),
        Vocabulary(target_tokenizer.tokens),
        validation_fraction=0.0,
        test_fraction=0.0,
        shuffle_buffer_size=2,
    )

    examples = list(dataset)

    assert len(examples) == 3
    assert all(source.ndim == target.ndim == 1 for source, target in examples)
