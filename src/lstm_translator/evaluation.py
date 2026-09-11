"""Standard corpus-level translation metrics."""

from collections.abc import Sequence

from sacrebleu.metrics import BLEU, CHRF


def translation_scores(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    """Calculate comparable BLEU and chrF scores for aligned translations."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    if not hypotheses:
        raise ValueError("at least one translation is required")
    reference_streams = [list(references)]
    return {
        "bleu": BLEU().corpus_score(list(hypotheses), reference_streams).score,
        "chrf": CHRF().corpus_score(list(hypotheses), reference_streams).score,
    }
