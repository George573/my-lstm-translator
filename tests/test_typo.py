import random
import pytest

from lstm_translator import TypoGenerator, TranslationDataset, Vocabulary


def test_probability_zero_and_none_are_clean():
    assert TypoGenerator(backend="none", corruption_probability=1)("clean") == "clean"
    assert TypoGenerator(backend="augly", corruption_probability=0)("clean") == "clean"


def test_backend_is_called_only_after_sentence_sampling():
    calls = []

    def fake(text, count):
        calls.append((text, count))
        return text + "!"

    generator = TypoGenerator(corruption_probability=1, rng=random.Random(4), augmenter=fake)
    assert generator("hello") == "hello!"
    assert calls and calls[0][1] >= 1


def test_longer_sentences_receive_more_errors_on_average():
    short_counts = []
    long_counts = []
    short = "a" * 20
    long = "a" * 200
    short_generator = TypoGenerator(corruption_probability=1, rng=random.Random(1),
                                    augmenter=lambda text, count: short_counts.append(count) or text)
    long_generator = TypoGenerator(corruption_probability=1, rng=random.Random(2),
                                   augmenter=lambda text, count: long_counts.append(count) or text)
    for _ in range(500):
        short_generator(short)
        long_generator(long)
    assert sum(long_counts) / len(long_counts) > sum(short_counts) / len(short_counts)
    assert max(short_counts) >= 2
    assert max(long_counts) <= 10


def test_empty_and_invalid_rate():
    generator = TypoGenerator(corruption_probability=1, augmenter=lambda text, count: "changed")
    assert generator("... 123") == "... 123"
    with pytest.raises(ValueError, match="typo_rate"):
        TypoGenerator(typo_rate=0)


def test_dataset_augments_source_but_not_target():
    class Tokenizer:
        def encode(self, text):
            return [text]

    vocabulary = Vocabulary(["hello", "bonjour"])
    source = [["hello"]]
    target = [["bonjour"]]
    dataset = TranslationDataset(
        source, target, vocabulary, vocabulary,
        source_texts=["hello"],
        source_tokenizer=Tokenizer(),
        source_typo_generator=TypoGenerator(
            corruption_probability=1, augmenter=lambda text, count: "changed",
        ),
    )
    encoded_source, encoded_target = dataset[0]
    assert encoded_source.tolist() == [vocabulary.token_to_index["<UNK>"], vocabulary.token_to_index["<EOS>"]]
    assert encoded_target.tolist() == [vocabulary.token_to_index["<SOS>"], vocabulary.token_to_index["bonjour"], vocabulary.token_to_index["<EOS>"]]
