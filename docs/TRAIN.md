# Training

## Language config

A language is one YAML file: its letters, its phonemes, how to normalize, which encoder, which data.

```yaml
# configs/sk.yaml
graphemes: aáäbcčdďeéfghiíjklĺľmnňoóôpqrŕsštťuúvwxyýzž
phonemes: abcdefijklmnoprstuvxzŋɟɡɣɦɱɲʃʎʒʣʤʦʧˈː̯̩
normalizers: [lowercase]
backbone: gerulata/slovakbert
train: data/sk/train.tsv
eval: data/sk/eval-silver.tsv
```

| key | |
|---|---|
| `graphemes` | letters that form words; everything else passes through |
| `phonemes` | every symbol a target may use, stress (`ˈ`) included |
| `normalizers` | applied in order: `strip_marks` (drop combining marks, e.g. niqqud), `lowercase` |
| `backbone` | any Hugging Face model with a fast tokenizer, or `none` for characters only |
| `train` | TSV path or glob, or a list of them |
| `eval` | one TSV |

Build `phonemes` from the labels you actually have: a target with a letter outside it is masked.

## Data

Plain TSV, `text<TAB>phonemes`, one phoneme token per whitespace token, punctuation left in place:

```text
Na stole je kniha.	nˈa stole je kɲˈiɦa.
```

A word trains only when both sides line up; otherwise it is masked and still serves as context:

- token counts differ → the whole row is masked
- a token's word counts differ, or its phonemes hold a letter outside `phonemes` → that token is masked

## Run

```bash
uv run accelerate launch --mixed_precision bf16 -m grafon.train --language configs/sk.yaml --output runs/sk-base
```

Every 10 minutes it evaluates and writes `runs/sk-base/last`; `best` follows the lowest eval loss. TensorBoard logs go to the same directory.

Stop with Ctrl-C (it checkpoints first) and continue with `--resume runs/sk-base/last`.

## Live access

The trainer listens for [pyinject](https://github.com/thewh1teagle/pyinject), so a running job can be inspected without stopping it:

```bash
uv run pyinject <pid> 'checkpoint(); print(state["step"])'
```
