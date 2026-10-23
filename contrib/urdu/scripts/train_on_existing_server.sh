#!/usr/bin/env bash
set -euo pipefail
# These paths are explicit to avoid coupling the upstream Grafon code to older research projects.
: "${GRAFON_TRAIN_TSV:?Set GRAFON_TRAIN_TSV to the approved training TSV}"
: "${GRAFON_VALIDATION_TSV:?Set GRAFON_VALIDATION_TSV to the validation TSV}"
: "${GRAFON_TEST_TSV:?Set GRAFON_TEST_TSV to the untouched test TSV}"
TRAIN="$GRAFON_TRAIN_TSV"
VAL="$GRAFON_VALIDATION_TSV"
TEST="$GRAFON_TEST_TSV"
for filename in "$TRAIN" "$VAL" "$TEST"; do
  if [[ ! -f "$filename" ]]; then echo "Missing data: $filename" >&2; exit 2; fi
done
python scripts/check_dataset.py --train "$TRAIN" --validation "$VAL" --test "$TEST"
mkdir -p runs logs reports
python -m grafon_urdu.train \
  --train "$TRAIN" --validation "$VAL" \
  --out runs/urdu-compact-s42 \
  --epochs "${GRAFON_EPOCHS:-12}" --batch-size "${GRAFON_BATCH_SIZE:-16}" \
  --max-source 192 --max-target 256 --device cuda --bf16 \
  2>&1 | tee logs/urdu-compact-s42.log
python -m grafon_urdu.eval \
  --model runs/urdu-compact-s42/best \
  --tsv "$TEST" --device cuda \
  --output reports/urdu-compact-s42-test.json
python scripts/prepare_model_card.py --model-dir runs/urdu-compact-s42/best --eval reports/urdu-compact-s42-test.json
python scripts/publish_hf.py --model-dir runs/urdu-compact-s42/best
