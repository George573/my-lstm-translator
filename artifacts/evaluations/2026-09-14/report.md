# Intermediate checkpoint evaluation — 2026-09-14

Evaluated `model-32mil-param.latest.pth` (32,598,427 parameters, 103,503 steps) against `model-256-6.latest.pth` (53,503 steps). Both have the same architecture.

| Local corpus (300 pairs each) | Previous BLEU | New BLEU | Previous chrF | New chrF |
|---|---:|---:|---:|---:|
| en-fr-13-len-727-sampled.csv | 27.43 | 28.79 | 48.15 | 50.13 |
| Tatoeba data/raw/fra.txt | 17.59 | 19.23 | 42.14 | 44.87 |

The newer checkpoint improves both metrics on both samples. This is a descriptive comparison, without significance testing. It remains unreliable on some basic meanings: “This apple is rotten.” becomes “Cette pomme est robuste.”; “I'm waiting for my friend.” becomes “Je suis attendu à mon ami.” It also translates some sentences correctly, including “People change.” and “Tom is in Boston with Mary.”

Saved validation loss fell from 9.5165 at interval 6 to 8.5514 at interval 11, with consecutive improvements from interval 8 onward. Training loss rose from 1.9747 at interval 7 to 2.1455 at interval 11 while teacher forcing fell from 0.9680 to 0.8656. Those training losses are not directly comparable because the conditioning changed. Validation uses zero teacher forcing in the current code, so its scale is also not directly comparable to training loss. The observed validation trend supports continued progress; it does not establish the optimal stopping point.

Method: deduplicate exact pairs per corpus, retain the repository's hash-based test partition (seed 42, test fraction 0.01, validation fraction 0.01), then sample 300 pairs per corpus with Python Random(42). CPU greedy decoding, batches of 16, maximum 100 generated tokens, embedded checkpoint tokenizers and vocabularies. The new model hit the generation limit on 1/300 examples in each corpus; the old model did so on 1/300 CSV examples and 0/300 Tatoeba examples. Raw references and default repository SacreBLEU metrics were used; signatures and every prediction are in results.json.

Limits: the original 21.4-million-pair training partition is unavailable here. These are local test-partition samples, not a verified reconstruction of the original training run's test set. Held-out status depends on the training run using the same partition logic/settings; source-only or near-duplicate overlap is not checked. The CSV includes some visibly mismatched references. Single-reference scoring also penalizes valid alternative translations.

Reproduce from the repository root:

```bash
.venv-3-12/bin/python artifacts/evaluations/2026-09-14/evaluate.py
```
