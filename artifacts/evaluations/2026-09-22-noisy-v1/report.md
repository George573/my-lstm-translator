# Noisy v1 model evaluation — 2026-09-22

Evaluated both checkpoints in `/home/unicorn/new_model/checkpoints`. Their model weights are tensor-identical and every prediction matches. Both embedded tokenizers exactly match `artifacts/tokenizers/noisy-v1/en.json` and `fr.json`; these tokenizer states were used for inference. File hashes and checks are in `provenance.json`.

The model has 47,735,004 parameters despite the `100mil` filename: six LSTM layers, hidden size 300, embedding dimension 84, and attention. The best checkpoint reports epoch 11 and validation loss 1.72385. The latest reports step 52,753. Teacher forcing stayed at 1.0 throughout the saved history, so validation loss measures teacher-forced prediction rather than free-running translation quality.

## Comparison on the historical examples

300 sentence pairs per corpus, exactly matching the prior report. Previous model: `model-32mil-param-quick-tf-decay.latest.pth`.

| Corpus | Previous BLEU | New BLEU | Change | Previous chrF | New chrF | Change |
|---|---:|---:|---:|---:|---:|---:|
| en-fr-13-len-727-sampled.csv | 30.27 | 29.56 | -0.71 | 50.97 | 49.88 | -1.10 |
| data/raw/fra.txt | 21.37 | 17.92 | -3.45 | 46.19 | 43.82 | -2.37 |

The new model scores lower on both metrics in both corpora. This small comparison does not establish statistical significance or isolate architecture, tokenizer, training-data, or training-duration effects. One output per corpus reached the 100-token limit. All predictions are in `comparison.json`.

## Fresh source-grouped test sample

Seed 42, test fraction 0.01, normalized-source partitioning, 300 pairs per corpus. These are different examples from the historical comparison, so their scores should not be compared directly with the previous model’s historical scores.

| Corpus | BLEU | chrF | Outputs at length limit |
|---|---:|---:|---:|
| en-fr-13-len-727-sampled.csv | 27.56 | 47.16 | 0 |
| data/raw/fra.txt | 23.28 | 46.27 | 0 |

All predictions are in `source-test.json`. Both evaluation sets contain clean source text; noisy-v1 refers to the tokenizer version, not an injected-typo benchmark.

## Method and limitations

CPU greedy autoregressive decoding, evaluation mode, four threads, batch size 16, maximum 100 generated tokens. SacreBLEU 2.6.0, case-sensitive BLEU with 13a tokenization and default chrF2. Full signatures are in the JSON files. Checkpoint vocabularies preserve the trained token indices. Training overlap has not been independently verified; source partitioning alone does not prove these external checkpoints never saw the samples. Single references penalize valid alternatives, some local references are imperfect, and the corpora may overlap. The full 8 GB training corpus was not evaluated.

## Reproduce

Run from the repository root:

```bash
.venv-3-12/bin/python scripts/evaluate.py \
  --checkpoints /home/unicorn/new_model/checkpoints/model-100mil.pth /home/unicorn/new_model/checkpoints/model-100mil.latest.pth \
  --examples-from artifacts/evaluations/2026-09-14-quick-tf-decay/results.json \
  --output artifacts/evaluations/2026-09-22-noisy-v1/comparison.json

.venv-3-12/bin/python scripts/evaluate.py \
  --checkpoints /home/unicorn/new_model/checkpoints/model-100mil.pth /home/unicorn/new_model/checkpoints/model-100mil.latest.pth \
  --data en-fr-13-len-727-sampled.csv data/raw/fra.txt \
  --sample-size 300 --seed 42 --test-fraction 0.01 \
  --output artifacts/evaluations/2026-09-22-noisy-v1/source-test.json
```
