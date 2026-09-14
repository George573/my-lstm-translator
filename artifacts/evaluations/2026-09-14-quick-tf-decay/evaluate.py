import json, random, time
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from sacrebleu.metrics import BLEU, CHRF
from lstm_translator.checkpoint import load_checkpoint
from lstm_translator.data import iter_parallel_rows, parallel_partition
from lstm_translator.model import EOS_INDEX, PAD_INDEX, SOS_INDEX

torch.set_num_threads(4)
out = Path(__file__).parent
previous = json.loads(Path('artifacts/evaluations/2026-09-14/results.json').read_text())
datasets = {path: [(e['source'], e['reference']) for e in scores['examples']] for path, scores in previous['models']['model-32mil-param.latest.pth']['datasets'].items()}
results = {'seed':42, 'sample_per_corpus':300, 'max_length':100, 'device':'cpu', 'note':'Local corpora, test hash partition using seed 42 and test fraction .01. Original training corpus unavailable; training overlap cannot be fully verified.', 'models':{}}
for name in ['model-32mil-param.latest.pth', 'model-32mil-param-quick-tf-decay.latest.pth']:
    bundle = load_checkpoint('artifacts/checkpoints/' + name)
    state = bundle.training_state
    result = {'metadata':bundle.metadata, 'parameters':sum(p.numel() for p in bundle.model.parameters()), 'global_step':state['global_step'], 'history':state['history'], 'teacher_forcing_ratio':state['teacher_forcing_ratio'], 'datasets':{}}
    bundle.training_state = None
    del state
    for path, pairs in datasets.items():
        start = time.monotonic()
        hypotheses, capped = [], 0
        for offset in range(0, len(pairs), 16):
            batch = pairs[offset:offset+16]
            tensors = [torch.tensor(bundle.source_vocabulary.encode(bundle.source_tokenizer.encode(s)) + [EOS_INDEX]) for s, _ in batch]
            with torch.inference_mode():
                generated = bundle.model.generate(pad_sequence(tensors, batch_first=True, padding_value=PAD_INDEX), torch.tensor([len(t) for t in tensors]), max_length=100)
            for row in generated.tolist():
                capped += EOS_INDEX not in row
                tokens = [i for i in row if i not in {EOS_INDEX, PAD_INDEX, SOS_INDEX}]
                hypotheses.append(bundle.target_tokenizer.decode(bundle.target_vocabulary.decode(tokens)))
        refs = [t for _, t in pairs]
        bleu, chrf = BLEU(), CHRF()
        scores = {'bleu':bleu.corpus_score(hypotheses,[refs]).score, 'chrf':chrf.corpus_score(hypotheses,[refs]).score, 'bleu_signature':str(bleu.get_signature()), 'chrf_signature':str(chrf.get_signature()), 'capped':capped, 'seconds':time.monotonic()-start, 'examples':[{'source':s,'reference':r,'hypothesis':h} for (s,r),h in zip(pairs,hypotheses)]}
        result['datasets'][path] = scores
        print(name, path, {k:v for k,v in scores.items() if k!='examples'}, flush=True)
    results['models'][name] = result
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    del bundle
