"""Run sequential CPU timing comparisons on identical saved source texts."""
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from lstm_translator.inference import Translator
from lstm_translator.model import EOS_INDEX, PAD_INDEX, SOS_INDEX

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent

def batch_translate(bundle, texts):
    tensors = [torch.tensor(bundle.source_vocabulary.encode(bundle.source_tokenizer.encode(text)) + [EOS_INDEX]) for text in texts]
    with torch.inference_mode():
        rows = bundle.model.generate(pad_sequence(tensors, batch_first=True, padding_value=PAD_INDEX), torch.tensor([len(t) for t in tensors]), max_length=100).tolist()
    return [bundle.target_tokenizer.decode(bundle.target_vocabulary.decode([i for i in row if i not in {EOS_INDEX, PAD_INDEX, SOS_INDEX}])) for row in rows]

def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    saved = json.loads((OUT / 'comparison.json').read_text())
    datasets = next(iter(saved['models'].values()))['datasets']
    sources = [item['source'] for items in zip(*(s['examples'][:64] for s in datasets.values())) for item in items]
    results = {'device':'cpu', 'cpu':'Intel Core i7-8700', 'torch':torch.__version__, 'threads':4, 'batch_size':16, 'max_length':100, 'sources':sources, 'note':'Sequential model runs; warm filesystem cache possible. Batch timing includes tokenization, generation and decoding, excludes metric scoring. Single requests use the public Translator.translate API. Three warm batch passes of 128 sources; 60 single requests after warmup.', 'models':{}}
    for path in [ROOT/'artifacts/checkpoints/model-32mil-param-quick-tf-decay.latest.pth', Path('/home/unicorn/new_model/checkpoints/model-100mil.pth')]:
        started=time.perf_counter()
        translator=Translator.from_checkpoint(str(path))
        load_seconds=time.perf_counter()-started
        started=time.perf_counter(); first=translator.translate(sources[0]); first_seconds=time.perf_counter()-started
        for text in sources[:5]: translator.translate(text)
        latencies=[]; examples=[]
        for text in sources[:60]:
            started=time.perf_counter(); output=translator.translate(text); latencies.append(time.perf_counter()-started)
            examples.append({'source':text,'hypothesis':output})
        batch_translate(translator.bundle,sources[:16])
        passes=[]
        for repeat in range(3):
            started=time.perf_counter()
            for offset in range(0,len(sources),16): batch_translate(translator.bundle,sources[offset:offset+16])
            passes.append(time.perf_counter()-started)
        result={'load_seconds':load_seconds,'first_request_seconds':first_seconds,'single_mean_ms':statistics.mean(latencies)*1000,'single_median_ms':statistics.median(latencies)*1000,'single_p95_ms':sorted(latencies)[56]*1000,'single_seconds':latencies,'batch_pass_seconds':passes,'batch_sentences_per_second':len(sources)/statistics.median(passes),'examples':examples}
        results['models'][path.name]=result
        (OUT/'runtime.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        print(path.name,{k:v for k,v in result.items() if k not in {'examples','single_seconds'}},flush=True)
        del translator; gc.collect()

if __name__=='__main__': main()
