"""Create HF model card from *measured* training/evaluation artifacts only."""
import argparse
import json
from pathlib import Path

TEMPLATE = '''---
language:
- ur
tags:
- grafon
- g2p
- grapheme-to-phoneme
- ipa
- urdu
- speech-synthesis
library_name: pytorch
---

# Grafon Urdu — Compact Grapheme-to-IPA Model

A compact, character/grapheme-level Urdu grapheme-to-IPA encoder-decoder Transformer.
Trained from scratch without BERT/XLM-R/LLM weights.

## Model architecture

- Parameters: **{parameters:,}**
- FP32 weight file: **{size:.2f} MB**
- Transformer model width: {width}, heads: {heads}, encoder layers: {encoder}, decoder layers: {decoder}
- Vocabulary: Urdu Unicode grapheme input, IPA Unicode grapheme output (including punctuation)
- Training unit: **{training_unit}**; context radius: **{radius}**
- Source data SHA256: `{train_hash}`
- Validation data SHA256: `{val_hash}`

## Evaluation

Reported results are from the legacy held-out test corpus **and are not an independently
adjudicated gold benchmark**. The source corpus contains repeated phrase templates and
limited vocabulary. This checkpoint has not been validated for general Urdu lexical
coverage or dialectal pronunciation variation.

- Sentences evaluated: {sentences}
- Exact-word agreement: **{accuracy:.3%}**
- IPA Unicode grapheme edit rate: **{ger:.3%}**
- Note: grapheme edit rate is **not** independently tokenized phonemic PER.
- Word-boundary mismatch sentences: **{boundary_mismatches}** (in context-word mode, alignment is guaranteed only for successful decoding).

## Inference

```python
from grafon_urdu import Phonemizer
model = Phonemizer.from_pretrained("grafon-g2p/ur", device="cpu")
print(model("مجھے اردو زبان سے محبت ہے۔"))
```

Install `grafon-urdu-g2p` from the Grafon source contribution (`contrib/urdu/`).

## Training

Training implementation: [Grafon](https://github.com/thewh1teagle/grafon)
(proposed Urdu contribution; link becomes applicable once merged). Original dataset
annotations involved human review, followed by documented automated mask repairs
and assistant-assisted QA. Do not treat downstream corrected rows as independent gold.

## Intended use and limitations

Speech synthesis preprocessing and Urdu pronunciation research. Errors are expected on
homographs, names, rare vocabulary, loanwords, punctuation, and regional pronunciations.
Manually inspect high-impact generated phonemes. This is a research model, not a certified
pronunciation dictionary.

## Provenance and license

Contributor and dataset licensing details require verification by the rights holder
prior to public publication. This model is not an official upstream Grafon release
unless accepted by its maintainers.
'''


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--eval', required=True)
    args=ap.parse_args()
    d=Path(args.model_dir)
    c=json.loads((d/'config.json').read_text(encoding='utf8'))
    train=json.loads((d/'training_report.json').read_text(encoding='utf8'))
    eval_report=json.loads(Path(args.eval).read_text(encoding='utf8'))
    required=('sentences','exact_word_accuracy','ipa_grapheme_error_rate')
    if not all(k in eval_report for k in required):
        raise SystemExit('Evaluation report missing required metrics')
    size=(d/'model.safetensors').stat().st_size/1e6
    content=TEMPLATE.format(parameters=train['parameter_count'],size=size,
        width=c['d_model'],heads=c['nhead'],encoder=c['encoder_layers'],decoder=c['decoder_layers'],
        train_hash=train['train_sha256'],val_hash=train['validation_sha256'],
        sentences=eval_report['sentences'],accuracy=eval_report['exact_word_accuracy'],
        ger=eval_report['ipa_grapheme_error_rate'],
        training_unit=train.get('training_unit', 'sentence'),
        radius=train.get('context_radius', 'N/A'),
        boundary_mismatches=eval_report.get('word_count_mismatch_sentences', 'not recorded'))
    (d/'README.md').write_text(content,encoding='utf8')
    print('Created',d/'README.md')

if __name__=='__main__':main()
