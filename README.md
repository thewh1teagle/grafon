# grafon

Contextual grapheme-to-phoneme for any language, on top of any text encoder.

A pretrained encoder reads the whole sentence; a small decoder spells out each word's phonemes. Context decides what spelling alone can't: Hebrew vowels, homographs, voicing across word boundaries.

```text
אחרי שאתה אורז את המזוודה, תבוא לאכול אורז  →  ʔaχʁˈej ʃeʔatˈa ʔoʁˈez ʔˈet hamizvadˈa, tavˈo leʔeχˈol ʔˈoʁez
Idem s mamou k otcovi v Prahe.              →  ˈiɟem z mˈamou̯ ɡ ˈotʦovi f prˈaɦe.
```

## Use

```bash
uv run grafon runs/sk-base/best "Ahoj svet, čo si myslíš o modeli?"
```

```python
from grafon import Phonemizer

phonemize = Phonemizer("runs/sk-base/best")
phonemize("Ahoj svet, čo si myslíš o modeli?")
```

Anything outside the language's letters (digits, punctuation, other scripts) passes through as written.

## Train

```bash
uv run accelerate launch --mixed_precision bf16 -m grafon.train --language configs/sk.yaml --output runs/sk-base
```

A language is one YAML file and the data is plain `text<TAB>phonemes`; see [Training](docs/TRAIN.md).

## Languages

| | encoder | data | eval |
|---|---|---|---|
| Hebrew | `dicta-il/neodictabert` | 6.0M sentences | 90.5% word accuracy |
| Slovak | `gerulata/slovakbert` | 9.5M sentences | training |
| Arabic | `UBC-NLP/MARBERTv2` | 1.0M sentences | training |
| English (US) | `jhu-clsp/ettin-encoder-150m` | 32.6M sentences + 14k homograph sentences | 97.5% homograph accuracy |

## Docs

- [Training](docs/TRAIN.md): language configs, data format, running and resuming
- [Architecture](docs/ARCHITECTURE.md): the model, the data format, decoding
- [Evaluation](docs/EVAL.md): MILIM-Bench for Hebrew, neurlang multi_eval for Slovak, Arabic Speech Corpus for Arabic
- [Project rules](docs/PROJECT.md): how training runs and model sizes are managed
