"""Re-evaluate the exact examples saved with this historical experiment."""
from pathlib import Path

from lstm_translator.evaluate_cli import main as evaluate

ROOT = Path(__file__).resolve().parents[3]


def main(argv=None):
    return evaluate(argv, defaults={
        "checkpoints": [ROOT / "artifacts/checkpoints" / name for name in
                        ("model-32mil-param.latest.pth", "model-32mil-param-quick-tf-decay.latest.pth")],
        "examples_from": Path(__file__).with_name("results.json"),
        "output": Path(__file__).with_name("rerun-results.json"),
    })


if __name__ == "__main__":
    raise SystemExit(main())
