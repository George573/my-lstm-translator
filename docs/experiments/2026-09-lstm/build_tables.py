"""Refresh report tables and the full translation appendix from saved outputs."""
import json
import re
from pathlib import Path

OUT = Path(__file__).parent
ORDER = ['early-53k', 'baseline-103k', 'decay-147k', 'decay-172k',
         'decay-best-epoch24', 'decay-224k', 'no-tf-named-228k',
         'expanded-10-layer', 'full-tf-noisy-v1']
LABELS = dict(zip(ORDER, ['Early 53k', 'Baseline 103k', 'Decay 147k (archive)', 'Decay 172k',
                        'Decay best, interval 24', 'Decay 224k', '“No-TF” file, 228k',
                        'Ten-layer continuation', 'Full TF, noisy-v1']))

def cell(text):
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('\n', '<br>')

def replace_section(text, name, body):
    start, end = f'<!-- {name} -->', f'<!-- END_{name} -->'
    replacement = start + '\n\n' + body + '\n\n' + end
    if end in text:
        return re.sub(re.escape(start)+r'.*?'+re.escape(end),lambda _: replacement,text,flags=re.S)
    assert start in text, name
    return text.replace(start,replacement)

def main():
    models = json.loads((OUT/'results.json').read_text())['models']
    assert set(ORDER) == set(models)
    model_rows = ['| Run | Layers / parameters | Saved steps | Training TF at snapshot | Validation policy |',
                  '|---|---:|---:|---:|---|']
    for key in ORDER:
        m=models[key]; layers=m['metadata']['model_config']['num_layers']
        tf=m['teacher_forcing_ratio']; ratio=f'{tf:.4f}' if tf is not None else '0.20 (interval-24 history)'
        steps=f"{m['global_step']:,}" if m['global_step'] is not None else 'Not saved'
        policy='TF=1 (explicit metadata)' if key=='full-tf-noisy-v1' else 'TF=0 (historical record)'
        model_rows.append(f"| {LABELS[key]} | {layers} / {m['parameters']/1e6:.2f}M | {steps} | {ratio} | {policy} |")
    scores=['| Run | CSV BLEU | CSV chrF | Tatoeba BLEU | Tatoeba chrF | Length-capped outputs (CSV / Tatoeba) |',
            '|---|---:|---:|---:|---:|---:|']
    for key in ORDER:
        a,b=(models[key]['datasets'][name] for name in ['en-fr-13-len-727-sampled.csv','data/raw/fra.txt'])
        scores.append(f"| {LABELS[key]} | {a['bleu']:.2f} | {a['chrf']:.2f} | {b['bleu']:.2f} | {b['chrf']:.2f} | {a['capped']} / {b['capped']} |")
    selected=[4,5,8,10,12,15,19]
    comparisons=[]
    for i in selected:
        source=models['full-tf-noisy-v1']['probes'][i]['source']
        comparisons += [f'**English: {cell(source)}**','', '| Model | Actual French output |', '|---|---|']
        for key in ['decay-147k','decay-224k','expanded-10-layer','full-tf-noisy-v1']:
            e=models[key]['probes'][i]; assert e['source']==source
            comparisons.append(f"| {LABELS[key]} | {cell(e['hypothesis'])} |")
        comparisons += ['']
    path=OUT/'README.md';text=path.read_text()
    for name,body in [('MODEL_TABLE','\n'.join(model_rows)),('SCORE_TABLE','\n'.join(scores)),('TRANSLATION_TABLES','\n'.join(comparisons))]:
        text=replace_section(text,name,body)
    path.write_text(text)
    lines=['# Actual translations: complete diagnostic comparison','',
           '[Return to the analysis](README.md). Generated on 22 September 2026 with each checkpoint’s embedded tokenizer and vocabulary. The full-TF tokenizer state matches noisy-v1. Outputs are unedited; markup characters are escaped only for Markdown rendering. Maximum generated length: 100 tokens. These are purposive probes, not a representative accuracy sample.','',
           'The first 15 prompts reproduce the recorded live examples; the new model’s outputs match that transcript. Prompt 16 completes the input for which the transcript did not include an output. The remaining prompts add clean/noisy counterparts and tests of factual details. Checkpoint identities and model hashes are in [results.json](results.json).','']
    for i, probe in enumerate(models['full-tf-noisy-v1']['probes'],1):
        lines += [f"## {i}. {cell(probe['source'])}",'','| Model | Actual output |','|---|---|']
        for key in ORDER:
            e=models[key]['probes'][i-1]; assert e['source']==probe['source']
            lines.append(f"| {LABELS[key]} | {cell(e['hypothesis'])} |")
        lines.append('')
    (OUT/'translations.md').write_text('\n'.join(lines))
    print('Updated report tables and all 29 probe comparisons.')

if __name__=='__main__': main()
