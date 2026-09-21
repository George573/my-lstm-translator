"""Sentence sampling with nlpaug's native keyboard augmentation policy."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator


class TypoGenerator:
    """Select sentences, then let KeyboardAug choose words and characters.

    ``rng`` controls sentence selection only. Seed Python and NumPy's global
    RNGs as well for reproducible nlpaug edits (see ``seed_everything``).
    A selected sentence may remain unchanged when no eligible edit exists.
    """

    def __init__(self, *, backend: str = "nlpaug", corruption_probability: float = 0.23,
                 aug_char_p: float = 0.1, aug_word_p: float = 0.1,
                 aug_char_max: int = 1, aug_word_max: int = 3,
                 lang: str = "en", rng: random.Random | None = None,
                 augmenter: Callable[[str], str | list[str]] | None = None) -> None:
        self.backend = backend.lower()
        if self.backend not in {"nlpaug", "none"}:
            raise ValueError("backend must be nlpaug or none")
        for name, value in (("corruption_probability", corruption_probability),
                            ("aug_char_p", aug_char_p), ("aug_word_p", aug_word_p)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name, value in (("aug_char_max", aug_char_max), ("aug_word_max", aug_word_max)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.corruption_probability = corruption_probability
        self.rng = rng if rng is not None else random
        self._augmenter = augmenter
        self.options = dict(aug_char_p=aug_char_p, aug_word_p=aug_word_p,
                            aug_char_min=1, aug_word_min=1,
                            aug_char_max=aug_char_max, aug_word_max=aug_word_max,
                            include_special_char=False, include_numeric=False, lang=lang)

    def __call__(self, text: str) -> str:
        if (self.backend == "none" or self.corruption_probability == 0
                or not any(char.isalpha() for char in text)
                or self.rng.random() >= self.corruption_probability):
            return text
        if self._augmenter is None:
            try:
                from nlpaug.augmenter.char import KeyboardAug
            except ImportError as exc:
                raise RuntimeError("typo augmentation requires nlpaug; install project dependencies "
                                   "or use --source-typo-backend none") from exc
            # One reusable backend per generator / DataLoader worker.
            self._augmenter = KeyboardAug(**self.options).augment
        result = self._augmenter(text)
        if isinstance(result, list):
            result = result[0] if result else text
        if not isinstance(result, str):
            raise TypeError("typo backend must return text or a list of texts")
        return result or text


def tokenizer_training_texts(texts: Iterable[str], generator: TypoGenerator) -> Iterator[str]:
    """Yield one text per example, replacing selected texts with typo variants.

    The generator's sentence probability governs which examples are changed.
    Input and output remain streaming and have the same number of examples.
    """
    for text in texts:
        noisy = generator(text)
        if noisy != text:
            yield noisy
        else:
            yield text
