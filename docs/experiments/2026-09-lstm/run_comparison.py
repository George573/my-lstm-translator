"""Reproduce the September checkpoint comparison from the repository root."""
import gc
import hashlib
import json
from pathlib import Path

import torch
from lstm_translator.checkpoint import load_checkpoint
from lstm_translator.evaluate_cli import evaluate_bundle
from lstm_translator.inference import Translator

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
# Each path identifies a particular saved experiment; identical basenames in
# different directories do not imply identical checkpoints.
MODELS = {
    'early-53k': Path.home() / 'checkpoints/model-256-6.latest.pth',
    'baseline-103k': Path.home() / 'checkpoints/model-32mil-param.latest.pth',
    'decay-147k': ROOT / 'artifacts/checkpoints/model-32mil-param-quick-tf-decay.latest.archive.pth',
    'decay-172k': ROOT / 'artifacts/checkpoints/model-32mil-param-quick-tf-decay.latest.pth',
    'decay-best-epoch24': Path.home() / 'checkpoints/model-32mil-param-quick-tf-decay.pth',
    'decay-224k': Path.home() / 'checkpoints/model-32mil-param-quick-tf-decay.latest.pth',
    'no-tf-named-228k': Path.home() / 'checkpoints/model-32mil-param-no-tf.latest.pth',
    'expanded-10-layer': Path.home() / 'checkpoints/model-expanded.latest.pth',
    'full-tf-noisy-v1': Path.home() / 'new_model/checkpoints/model-100mil.latest.pth',
}
PROBES = [
    'Hello', 'How are you?', 'Do u know how are you?',
    'Russia is the largest country in the world.',
    'A small mistake can cause another mistake.',
    'One wrong word can change the whole sentence.',
    'I think the socio-economic impact of heavy AI usage may become negative rather than positive.',
    'I think the socio-economic impact of heavy AI usage may be positive.',
    'The model sometimes repeats words when it makes a mistake.',
    'If the model predicts one wrong token, the next prediction may also become wrong.',
    'My name is John. I am commander of the northern army. I am a general and a loyal servant to the true king.',
    'i dont know where mike is', 'I hav a golden aple', 'sixseven AI startup', 'I knpw jon',
    'I know John, but John does not know me.',
    'I have a golden apple.', 'I know John.', "I don't know where Mike is.",
    'The price decreased from 50 euros to 30 euros.',
    'The price increased from 30 euros to 50 euros.',
    'I am waiting for my friend.', 'I am not waiting for my friend.',
    'The meeting starts at 9:30 tomorrow morning.',
    'If it rains tomorrow, we will stay at home.',
    'Please send me the report before Friday.', 'Please send me the reprot before Friday.',
    'Tom felt safe.', 'The Commission shall comprise nine members.',
]

def weight_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), help='Evaluate only these runs and retain other saved results.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reference = ROOT / 'artifacts/evaluations/2026-09-14-quick-tf-decay/results.json'
    previous = json.loads(reference.read_text())
    datasets = {name: [(e['source'], e['reference']) for e in score['examples']]
                for name, score in next(iter(previous['models'].values()))['datasets'].items()}
    result = {'method': {'device': 'cpu', 'torch': torch.__version__, 'threads': 4,
                        'batch_size': 16, 'max_length': 100, 'decoding': 'greedy',
                        'examples_from': str(reference.relative_to(ROOT)),
                        'sample_per_corpus': 300, 'training_overlap': 'unverified',
                        'probe_note': 'Purposive diagnostic examples, including user-supplied prompts; not a random test set.'},
              'models': {}}
    if args.models and (OUT/'results.json').exists():
        result = json.loads((OUT/'results.json').read_text())
    for label, path in MODELS.items():
        if args.models and label not in args.models:
            continue
        print('Loading', label, path, flush=True)
        bundle = load_checkpoint(path)
        state = bundle.training_state or {}
        entry = {'checkpoint': str(path), 'weight_sha256': weight_digest(bundle.model),
                 'parameters': sum(p.numel() for p in bundle.model.parameters()),
                 'metadata': bundle.metadata, 'global_step': state.get('global_step'),
                 'teacher_forcing_ratio': state.get('teacher_forcing_ratio'),
                 'history': state.get('history', []), 'settings': state.get('settings', {}),
                 'tokenizer_versions': {side: getattr(bundle, side+'_tokenizer').text_processing_version
                                        for side in ['source', 'target']}, 'datasets': {}}
        bundle.training_state = None
        del state
        if label == 'full-tf-noisy-v1':
            from lstm_translator.tokenizer import BPETokenizer
            for side, lang in [('source', 'en'), ('target', 'fr')]:
                assert getattr(bundle, side+'_tokenizer').to_dict() == BPETokenizer.load(ROOT/f'artifacts/tokenizers/noisy-v1/{lang}.json').to_dict()
        for name, pairs in datasets.items():
            entry['datasets'][name] = evaluate_bundle(bundle, pairs, 16, 100)
            scores = entry['datasets'][name]
            print(label, name, {k: scores[k] for k in ['bleu', 'chrf', 'capped']}, flush=True)
        translator = Translator(bundle)
        entry['probes'] = [{'source': source, 'hypothesis': translator.translate(source)} for source in PROBES]
        result['models'][label] = entry
        (OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        del translator, bundle
        gc.collect()

if __name__ == '__main__':
    main()
