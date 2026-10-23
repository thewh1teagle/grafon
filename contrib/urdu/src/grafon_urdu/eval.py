"""Held-out evaluation without embedding test data in model artifacts."""
from __future__ import annotations

import argparse
import json
from itertools import zip_longest
from pathlib import Path

from .infer import Phonemizer
from .text import graphemes, read_tsv, save_json


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(previous[j] + 1, row[-1] + 1,
                           previous[j-1] + (x != y)))
        previous = row
    return previous[-1]


def evaluate(ph: Phonemizer, rows: list[tuple[str, str]], limit=0):
    correct, count, edits, reference_length = 0, 0, 0, 0
    predictions = []
    selected = rows[:limit or None]
    for start in range(0, len(selected), 16):
      chunk = selected[start:start+16]
      outputs = ph.phonemize_many([text for text, _ in chunk])
      for (text, target), prediction in zip(chunk, outputs):
        words_ref, words_pred = target.split(), prediction.split()
        for a, b in zip_longest(words_ref, words_pred, fillvalue=""):
            correct += int(a == b and bool(a))
            count += 1
        edits += distance(graphemes(target), graphemes(prediction))
        reference_length += len(graphemes(target))
        predictions.append({"text": text, "reference": target, "prediction": prediction})
    mismatched_boundaries = sum(len(row["reference"].split()) != len(row["prediction"].split()) for row in predictions)
    return {"sentences": len(predictions), "words": count,
            "word_count_mismatch_sentences": mismatched_boundaries, "exact_word_accuracy": correct/max(1,count),
            "ipa_grapheme_error_rate": edits/max(1,reference_length),
            "metric_note": "Unicode grapheme edit rate; not phoneme PER"}, predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = read_tsv(args.tsv)
    metrics, predictions = evaluate(Phonemizer.from_pretrained(args.model, args.device), rows, args.limit)
    save_json(args.output, metrics)
    with Path(args.output).with_suffix(".jsonl").open("w", encoding="utf-8") as fp:
        for item in predictions:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
