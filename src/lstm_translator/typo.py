"""Runtime source-text typo augmentation backed by external libraries."""

from __future__ import annotations

import random
import math
from collections.abc import Callable


class TypoGenerator:
    """Sample sentence corruption and delegate typo creation to a backend."""

    def __init__(self, *, backend: str = "augly", corruption_probability: float = 0.23,
                 typo_rate: float = 0.015,
                 rng: random.Random | None = None,
                 augmenter: Callable[[str, int], str] | None = None) -> None:
        self.backend = backend.lower()
        self.corruption_probability = corruption_probability
        self.typo_rate = typo_rate
        self.rng = rng if rng is not None else random
        self._augmenter = augmenter
        self._validate()

    def __call__(self, text: str) -> str:
        if self.backend == "none" or self.rng.random() >= self.corruption_probability:
            return text
        count = self._sample_error_count(text)
        return text if count == 0 else self._augment(text, count)

    def _sample_error_count(self, text: str) -> int:
        eligible_length = sum(char.isalpha() for char in text)
        if eligible_length == 0:
            return 0
        lam = self.typo_rate * eligible_length
        count = self._sample_poisson(lam)
        while count == 0:
            count = self._sample_poisson(lam)
        max_errors = max(3, math.ceil(eligible_length * 0.05))
        return min(count, max_errors)

    def _sample_poisson(self, lam: float) -> int:
        """Sample Poisson(lam) with only the standard library RNG."""
        threshold = math.exp(-lam)
        probability = 1.0
        count = 0
        while probability > threshold:
            count += 1
            probability *= self.rng.random()
        return count - 1

    def _augment(self, text: str, count: int) -> str:
        if self._augmenter is not None:
            return self._augmenter(text, count)
        current = text
        for _ in range(count):
            for _ in range(12):
                candidate = self._augment_once(current)
                if candidate != current and self._edit_distance_at_most_one(current, candidate):
                    current = candidate
                    break
            else:
                break
        return current

    def _augment_once(self, text: str) -> str:
        if self.backend == "augly":
            return self._augment_augly(text, 1)
        if self.backend == "nlpaug":
            return self._augment_nlpaug(text, 1)
        raise ValueError(f"unknown typo backend: {self.backend!r}")

    @staticmethod
    def _edit_distance_at_most_one(source: str, candidate: str) -> bool:
        """Return whether two strings differ by at most one insertion/deletion/substitution."""
        if abs(len(source) - len(candidate)) > 1:
            return False
        previous = list(range(len(candidate) + 1))
        for source_index, source_char in enumerate(source, 1):
            current = [source_index]
            for candidate_index, candidate_char in enumerate(candidate, 1):
                current.append(min(
                    current[-1] + 1,
                    previous[candidate_index] + 1,
                    previous[candidate_index - 1] + (source_char != candidate_char),
                ))
            if min(current) > 1:
                return False
            previous = current
        return previous[-1] <= 1

    @staticmethod
    def _augment_augly(text: str, count: int) -> str:
        try:
            from augly.text import transforms
        except ImportError as exc:
            raise RuntimeError("backend 'augly' requires the 'augly' package; install the augmentation extra or use --source-typo-backend none") from exc
        transform = transforms.SimulateTypos(
            aug_char_p=0.2, aug_char_min=count, aug_char_max=count,
            aug_word_p=1.0, aug_word_min=1, aug_word_max=1,
            n=1, typo_type="all", p=1.0,
        )
        result = transform(text)
        return result[0] if isinstance(result, list) else result

    @staticmethod
    def _augment_nlpaug(text: str, count: int) -> str:
        try:
            from nlpaug.augmenter.char import KeyboardAug
        except ImportError as exc:
            raise RuntimeError("backend 'nlpaug' requires the 'nlpaug' package") from exc
        result = KeyboardAug(aug_char_p=0.2, aug_char_min=count, aug_char_max=count).augment(text)
        return result[0] if isinstance(result, list) else result

    def _validate(self) -> None:
        if self.backend not in {"augly", "nlpaug", "none"}:
            raise ValueError("backend must be one of: augly, nlpaug, none")
        if not 0 <= self.corruption_probability <= 1:
            raise ValueError("corruption_probability must be in [0, 1]")
        if not math.isfinite(self.typo_rate) or self.typo_rate <= 0:
            raise ValueError("typo_rate must be finite and greater than zero")
