# Quick teacher-forcing decay checkpoint comparison

Compared `model-32mil-param-quick-tf-decay.latest.pth` against `model-32mil-param.latest.pth`, rerunning both on the same 600 pairs as the earlier evaluation (300 per corpus). Both have 32,598,427 parameters.

| Corpus | Previous BLEU | New BLEU | Change | Previous chrF | New chrF | Change |
|---|---:|---:|---:|---:|---:|---:|
| Local CSV | 28.79 | 30.27 | +1.48 | 50.13 | 50.97 | +0.84 |
| Tatoeba | 19.23 | 21.37 | +2.14 | 44.87 | 46.19 | +1.32 |

BLEU increased by 5.1% and 11.1% relative, respectively; these are metric gains, not percentages of translations that are correct. Baseline scores reproduce the earlier evaluation exactly.

SacreBLEU paired bootstrap tests with 1,000 resamples and seed 42 give BLEU p-values 0.036 (CSV) and 0.031 (Tatoeba). chrF p-values are 0.072 and 0.063. These are unadjusted exploratory tests across four comparisons; evidence is encouraging but not conclusive. Scores improve on both corpora, with mixed changes on individual sentences.

Examples (illustrative, not a separate benchmark):

- “Tom knows who I am.”: “Tom sait que je suis.” → “Tom sait qui je suis.” Corrected.
- “This apple is rotten.”: “Cette pomme est robuste.” → “Cette pomme est ronde.” Still wrong: rotten should be “pourrie”.
- “Please remind me to write a letter tomorrow.”: “Veuillez me rappeler de rédiger une lettre demain.” → “Veuillez me rappeler que vous écrire une lettre demdemain.” A regression.

Training advanced from 103,503 to 147,006 steps (+43,503, or 42.0%). Saved teacher forcing fell from 0.8656 to 0.5994. Last saved validation loss fell from 8.5514 to 7.1577 (16.3%), with improvements at every recorded interval since the previous checkpoint. Training loss rose from 2.1455 to 2.8010, but changing teacher forcing makes these losses unsuitable for direct comparison. This comparison cannot isolate the effect of faster teacher-forcing decay from additional training.

Method: reuse exact source/reference pairs from ../2026-09-14/results.json; those pairs were sampled after exact-pair deduplication and hash test partitioning (seed 42, validation/test fractions 0.01), with Python Random(42). CPU greedy decoding, 4 threads, batches of 16, maximum 100 generated tokens, checkpoint-embedded vocabularies/tokenizers, evaluation mode. SacreBLEU 2.6.0, default BLEU and chrF2. Full metric signatures and all predictions are in results.json. The generation limit was reached on 1 CSV example for each model, and on 1 Tatoeba example for the previous model versus 0 for the new model.

Limits: the original training corpus is unavailable, so these samples are not independently verified held-out data. Some CSV references are mismatched, and single references penalize legitimate alternatives. The two local corpora may overlap; their results are not independent replications. No claim of production-level translation quality is supported by this small evaluation.

Reproduce from the repository root:

```bash
.venv-3-12/bin/python artifacts/evaluations/2026-09-14-quick-tf-decay/evaluate.py
.venv-3-12/bin/python artifacts/evaluations/2026-09-14-quick-tf-decay/analyze.py
```
