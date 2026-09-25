"""Score a checkpoint on MILIM-Bench with its own metric: per target word, both sides stripped
to phoneme characters (stress included), so punctuation and spacing never decide a match.

uv run python scripts/eval_milim.py runs/he-base/last
Writes <run>/milim-<checkpoint>-step<N>.tsv next to the checkpoint (a checkpoint directory
is replaced on every save) and prints per-category word accuracy.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from grafon import Phonemizer

PHONEMES = set("abdefhijklmnopstuvwzɡʁʃʒʔˈχ")


def clean(ipa):
    return "".join(c for c in ipa if c in PHONEMES)


def targets(label):
    return {int(i): clean(ipa) for i, _, ipa in (part.partition("=") for part in label.split() if "=" in part)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--gold", type=Path, default=Path("data/MILIM-Bench/data/gold.tsv"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--beam", type=int, default=1)
    args = p.parse_args()

    with args.gold.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    phonemize = Phonemizer(args.checkpoint, beam=args.beam)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["Text"]))
    predictions = [None] * len(rows)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=phonemize.device.type == "cuda"):
        for start in tqdm(range(0, len(order), args.batch_size), desc="MILIM", unit="batch"):
            chunk = order[start:start + args.batch_size]
            for i, out in zip(chunk, phonemize([rows[i]["Text"] for i in chunk])):
                predictions[i] = out

    scores = defaultdict(lambda: [0, 0])
    for row, prediction in zip(rows, predictions):
        tokens = prediction.split()
        for index, gold in targets(row["Label"]).items():
            said = clean(tokens[index]) if index < len(tokens) else ""
            for key in (row["Category"], "ALL"):
                scores[key][0] += said == gold
                scores[key][1] += 1

    trainer = args.checkpoint / "trainer.json"
    step = json.loads(trainer.read_text())["step"] if trainer.is_file() else "na"
    out = args.checkpoint.parent / f"milim-{args.checkpoint.name}-step{step}.tsv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Category", "Text", "Label", "Prediction"])
        writer.writerows([row["Category"], row["Text"], row["Label"], pred] for row, pred in zip(rows, predictions))
    width = max(map(len, scores))
    for key in sorted(scores, key=lambda k: (k == "ALL", k)):
        right, total = scores[key]
        print(f"{key:<{width}}  {100 * right / total:5.1f}%  ({right}/{total})")
    print(f"predictions: {out}")


if __name__ == "__main__":
    main()
