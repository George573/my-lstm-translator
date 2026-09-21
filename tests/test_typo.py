import random

import pytest

from lstm_translator import TypoGenerator, TranslationDataset, Vocabulary, seed_everything
from lstm_translator.typo import tokenizer_training_texts


def test_probability_zero_and_none_are_clean():
    def fail(text):
        raise AssertionError('backend called')
    assert TypoGenerator(backend='none', corruption_probability=1, augmenter=fail)('clean') == 'clean'
    assert TypoGenerator(corruption_probability=0, augmenter=fail)('clean') == 'clean'


def test_backend_called_once_and_native_output_accepted():
    calls = []
    def fake(text):
        calls.append(text)
        return ['multiple changes are accepted']
    generator = TypoGenerator(corruption_probability=1, augmenter=fake)
    assert generator('hello world') == 'multiple changes are accepted'
    assert calls == ['hello world']


def test_sentence_sampling():
    calls = []
    generator = TypoGenerator(corruption_probability=.23, rng=random.Random(42),
                              augmenter=lambda text: calls.append(text) or text)
    for _ in range(10000):
        generator('hello world')
    assert 2100 < len(calls) < 2500


@pytest.mark.parametrize('options', [dict(backend='augly'), dict(aug_char_p=float('nan')),
    dict(aug_word_p=2), dict(aug_char_max=0), dict(aug_word_max=1.5)])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        TypoGenerator(**options)


def test_empty_and_non_alphabetic_unchanged():
    generator = TypoGenerator(corruption_probability=1, augmenter=lambda text: 'changed')
    assert generator('... 123') == '... 123'
    assert generator('') == ''


def test_tokenizer_stream_replaces_selected_texts_without_adding_examples():
    generator = TypoGenerator(corruption_probability=1,
                             augmenter=lambda text: 'noisy' if text == 'clean' else text)
    assert list(tokenizer_training_texts(iter(['clean', 'unchanged']), generator)) == [
        'noisy', 'unchanged']


def test_real_backend_reuse_and_seeded_edits():
    def run():
        seed_everything(42)
        generator = TypoGenerator(corruption_probability=1)
        results = [generator('Keyboard augmentation changes familiar English words.') for _ in range(8)]
        backend = generator._augmenter
        generator('Another sentence for checking backend reuse.')
        assert generator._augmenter is backend
        return results
    first = run()
    assert first == run()
    assert any(text != 'Keyboard augmentation changes familiar English words.' for text in first)


def test_dataset_augments_source_but_not_target():
    class Tokenizer:
        def encode(self, text):
            return [text]
    vocabulary = Vocabulary(['hello', 'bonjour'])
    dataset = TranslationDataset([['hello']], [['bonjour']], vocabulary, vocabulary,
        source_texts=['hello'], source_tokenizer=Tokenizer(),
        source_typo_generator=TypoGenerator(corruption_probability=1, augmenter=lambda text: 'changed'))
    source, target = dataset[0]
    assert source.tolist() == [3, 2]
    assert target.tolist() == [1, vocabulary.token_to_index['bonjour'], 2]
