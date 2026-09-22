# English-to-French LSTM Translator

An educational PyTorch translation project with custom BPE tokenizers, a
bidirectional LSTM encoder, and an attention-based autoregressive decoder.
Includes training notebooks, checkpointing, typo augmentation, and BLEU/chrF evaluation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[notebooks]"
jupyter lab
```

Start with `notebooks/train_tokenizers.ipynb` and `notebooks/train_model.ipynb`.
Use `notebooks/training_dashboard.ipynb` to inspect training histories.
For command-line training, run `python scripts/train.py --help`.

Translate with a trained checkpoint:

```python
from lstm_translator import Translator

translator = Translator.from_checkpoint("artifacts/checkpoints/model-v1.pth")
print(translator.translate("A small mistake can cause another mistake."))
```

## Experiment findings

Early teacher-forcing decay gave modest gains; later low-TF and deeper runs
regressed despite falling validation loss. The new full-TF model produces
cleaner sentences on many prompts, but still changes meanings and drops details.
Sequential attention and recurrent decoding make training expensive; adding
layers increases that cost without guaranteeing better translations.

- [Experiment analysis and training graphs](docs/experiments/2026-09-lstm/README.md)
- [Actual translations across nine checkpoints](docs/experiments/2026-09-lstm/translations.md)
- [Setup, training, tokenizers, evaluation, and troubleshooting](docs/usage.md)

This is an experimental model, not a production translation service.
Software: [GPL-2.0](LICENSE). Dataset attribution and licensing are recorded separately.
