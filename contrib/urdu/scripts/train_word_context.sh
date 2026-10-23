#!/usr/bin/env bash
set -euo pipefail
: "${GRAFON_TRAIN_TSV:?Missing training TSV}"
: "${GRAFON_VALIDATION_TSV:?Missing validation TSV}"
: "${GRAFON_TEST_TSV:?Missing test TSV}"
mkdir -p runs reports logs
# Never overwrite old sentence-level checkpoint.
RUN="runs/urdu-word-context-s42"
python -m grafon_urdu.train \
  --train "$GRAFON_TRAIN_TSV" --validation "$GRAFON_VALIDATION_TSV" \
  --out "$RUN" \
  --training-unit word_context --context-radius 2 \
  --epochs "${GRAFON_EPOCHS:-8}" --batch-size "${GRAFON_BATCH_SIZE:-128}" \
  --max-val-batches 0 --max-source 192 --max-target 128 \
  --device cuda --bf16 \
  2>&1 | tee logs/urdu-word-context-s42.log
python -m grafon_urdu.eval \
  --model "$RUN/best" --tsv "$GRAFON_TEST_TSV" --device cuda \
  --output reports/urdu-word-context-s42-test.json
python scripts/prepare_model_card.py --model-dir "$RUN/best" \
  --eval reports/urdu-word-context-s42-test.json
python scripts/publish_hf.py --model-dir "$RUN/best"
