import json, os
from pathlib import Path
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest
os.environ['SACREBLEU_SEED'] = '42'
out = Path(__file__).parent
r = json.loads((out/'results.json').read_text())
a,b = r['models'].values()
results = {}
for path in a['datasets']:
    old,new = a['datasets'][path], b['datasets'][path]
    assert [(e['source'],e['reference']) for e in old['examples']] == [(e['source'],e['reference']) for e in new['examples']]
    signatures, scores = PairedTest([('previous',[e['hypothesis'] for e in old['examples']]),('quick_tf_decay',[e['hypothesis'] for e in new['examples']])], {'BLEU':BLEU(),'chrF':CHRF()}, [[e['reference'] for e in old['examples']]], test_type='bs', n_samples=1000)()
    results[path] = {'signatures':{k:str(v) for k,v in signatures.items()},'scores':{k:[vars(x) for x in v] for k,v in scores.items() if k != 'System'}}
    print(path, results[path], flush=True)
(out/'significance.json').write_text(json.dumps(results,indent=2,default=float))
