# Grafon Urdu (ur) — compact contextual G2P contribution

This directory is a **proposed contribution** to [thewh1teagle/grafon](https://github.com/thewh1teagle/grafon), not an upstream release or a replacement for the existing Grafon implementation. No upstream API compatibility is claimed until maintainers review its integration. The implementation is self-contained and installable inside a Grafon clone via `pip install -e contrib/urdu`.

## Goal

Train a genuinely small Urdu grapheme-to-IPA model using a character/grapheme encoder-decoder Transformer. The preferred release candidate uses context-aware per-word decoding to avoid word boundary drift. No XLM-R or BERT checkpoint is packaged. The measured word-context checkpoint has 1,713,728 parameters and 6.86 MB FP32 weights. It supports sentence context and preserves Urdu word boundaries in targets. A word-context checkpoint has been trained and evaluated on an in-domain 4,996-sentence test set; an independently adjudicated gold benchmark remains pending.

## Install

```bash
python -m pip install -e 'contrib/urdu[hub,test]'
python -m pytest -q contrib/urdu/tests
```

For a standalone release directory, use `python -m pip install -e '.[hub,test]'`.

## Data format

UTF-8 TSV with exactly two columns, one sentence and one IPA sentence, tab-separated, without a header. A row's source words and IPA space-delimited words must align. All `[MASK]` annotations are refused. The supervised corpus may have reviewed source labels plus documented silver repair proposals; this is **not** automatically an independent gold corpus.

Keep original reviewed labels and downstream automated corrections distinguished in the dataset card. Do not include gold benchmark candidates in training.

## Train

```bash
python -m grafon_urdu.train \
  --train /absolute/path/to/train.tsv \
  --validation /absolute/path/to/validation.tsv \
  --out runs/urdu-word-context-s42 \
  --training-unit word_context --context-radius 2 \
  --epochs 8 --batch-size 128 --max-val-batches 0 --bf16 --device cuda
```

If CUDA memory is tight reduce `--batch-size` to 8; for a smoke test set `--max-train-rows 32 --epochs 1` and a distinct output. No data is silently truncated. `best/` is selected by validation loss. On the current 4,996-sentence in-domain test, the word-context model achieved 95.75% exact-word agreement and 4.246% reference-normalized word error rate. These are not independently adjudicated gold results.

## Inference

```python
from grafon_urdu import Phonemizer
phonemizer = Phonemizer.from_pretrained('runs/urdu-word-context-s42/best', device='cpu')
print(phonemizer('مجھے اردو زبان سے محبت ہے۔'))
```

After an authorized Hugging Face release replace the local path with `grafon-g2p/ur`.

```bash
grafon-urdu --model runs/urdu-word-context-s42/best --device cpu 'مجھے اردو زبان سے محبت ہے۔'
python -m grafon_urdu.eval --model runs/urdu-word-context-s42/best \
  --tsv /absolute/path/to/test.tsv --output reports/test_metrics.json
```

## Hub publication

`python scripts/publish_hf.py --model-dir runs/urdu-word-context-s42/best --repo grafon-g2p/ur` is a read-only dry run. `--execute` checks for organization membership and refuses a checkpoint over 60 MB. Uploads always start private. This publisher cannot grant you write access to `grafon-g2p`; the owner must authorize the contribution. It does not delete or overwrite an existing model unless the contributor explicitly allows an existing repo.

## Known limitations

- Training vocabulary in the available 40,361-row historical corpus is narrow and template repetition is high.
- Sentence-level IPA targets include punctuation; this is a grapheme-level output, not a curated phoneme inventory.
- The architecture and new model are unvalidated at deployment quality; the former 98.745% diagnostic result belongs to a **different, 280M-parameter teacher** and must not be attributed to this model.
- Full CUDA training and an in-domain evaluation have completed; independent lexical gold evaluation is still outstanding.
- Distribution rights for the original annotations and upstream code must be confirmed before a public upload.

## Contributing upstream

Fork `thewh1teagle/grafon`, add this directory under `contrib/urdu`, run the tests and open a pull request. Maintainers decide whether to integrate this adapter into Grafon's native registry and API. This package does not claim to control Grafon or its Hugging Face organization.

Before a release, generate the model card from real test metrics:

```bash
python scripts/prepare_model_card.py \
  --model-dir runs/urdu-word-context-s42/best \
  --eval reports/urdu-compact-s42-test.json
python scripts/publish_hf.py \
  --model-dir runs/urdu-word-context-s42/best \
  --repo grafon-g2p/ur
```

The uploader performs a dry run by default. With proper organization privileges and confirmed rights, `--execute --rights-confirmed` stages privately. To prevent accidental modifications, it refuses to overwrite existing models unless `--allow-existing` is supplied, and it will not overwrite an existing public repo.

Dataset publication, if authorized, is optional using `scripts/publish_dataset.py` and requires rights confirmation. Avoid representing downstream machine-assisted corrections as independently reviewed gold labels.

## Verified in-domain research result (2026-10-23)

Word-context model: 1,713,728 parameters / 6.86 MB FP32 weights; 95.75% exact-word agreement; 4.246% reference-normalized word error rate; 3,234/4,996 exact sentences; 1.408% IPA Unicode grapheme edit rate. These are in-domain mixed-provenance reference results, **not independently adjudicated gold**. See root `UPSTREAM_PR.md` for the comparison. Do not conflate legacy exact-word agreement with word-edit accuracy.
