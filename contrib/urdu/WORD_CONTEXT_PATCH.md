# Grafon Urdu v0.2: context-aware word decoding

This is an **experimental, independent contribution** and not an upstream Grafon release.
The 6.86 MB sentence-level model remains untouched. Word-context training
initializes a fresh compact model of similar size. Do not publish until
comparing held-out metrics.

## Why

The previous greedy sentence decoder frequently deletes, inserts, or changes
word boundaries (1,484/4,996 test sentences), sometimes hallucinating IPA for
other words. v0.2 marks one Urdu word with `⟦` and `⟧`, includes two words of
left/right context, and predicts only the marked word's IPA. Inference
reassembles one IPA token per source word. This guarantees boundary counts
when decoding completes successfully; pronunciation accuracy is not guaranteed.

## Apply patch

Extract this patch ZIP over the **existing**
`grafon-urdu-contribution/contrib/urdu` directory; see PATCH_QUICKSTART.md.

## Test first

`python -m pytest ./tests -q`

## Training

Set GRAFON_TRAIN_TSV, GRAFON_VALIDATION_TSV, GRAFON_TEST_TSV as before,
then `bash scripts/train_word_context.sh`.

For a small pilot:

```
python -m grafon_urdu.train --train "$GRAFON_TRAIN_TSV" \
 --validation "$GRAFON_VALIDATION_TSV" --out runs/word-pilot \
 --training-unit word_context --context-radius 2 \
 --max-train-rows 500 --epochs 1 --batch-size 64 \
 --max-val-batches 2 --max-source 192 --max-target 128 --device cuda --bf16
python -m grafon_urdu.eval --model runs/word-pilot/best \
 --tsv "$GRAFON_TEST_TSV" --limit 50 --device cuda \
 --output reports/word-pilot-test.json
```

Accuracy notes: The corpus has a narrow normalized Urdu vocabulary and
significant template repetition; 4,996 held-out sentences alone do not prove
SOTA. The original data have mixed reviewed/silver pronunciation provenance.
