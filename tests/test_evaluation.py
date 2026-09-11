import pytest

from lstm_translator import translation_scores


def test_identical_translations_score_perfect_chrf():
    scores = translation_scores(["bonjour le monde"], ["bonjour le monde"])
    assert scores["chrf"] == pytest.approx(100.0)
