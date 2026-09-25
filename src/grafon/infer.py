"""Inference: text → phonemes; everything outside the grapheme inventory passes through.

uv run grafon runs/he-base/best "שלום עולם"      # or lines on stdin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


class Phonemizer:
    def __init__(self, checkpoint, device=None, beam=1):
        from transformers import AutoTokenizer
        from .data import Collate
        from .model import G2P
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = G2P.from_checkpoint(checkpoint).to(self.device).eval()
        self.beam = beam
        s = self.model.settings
        tokenizer = None if s["backbone"] in (None, "none") else AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.collate = Collate(self.model.language, tokenizer, max_chars=s["max_chars"], max_tokens=s["max_tokens"],
                               max_word_chars=s["max_word_chars"], max_phonemes=s["max_phonemes"])

    def __call__(self, texts):
        from .data import assemble
        single = isinstance(texts, str)
        rows = self.collate.rows([(t, None) for t in ([texts] if single else texts)])
        batch = self.collate.encode(rows)
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        predicted = self.model.generate(batch, beam=self.beam)
        per_row = [[] for _ in rows]
        for row, ids in zip(batch["word_rows"], predicted):
            per_row[row].append(self.model.language.decode(ids))
        out = [assemble(row["text"], row["spans"], phonemes) for row, phonemes in zip(rows, per_row)]
        return out[0] if single else out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("text", nargs="*", help="Text to phonemize; stdin lines when omitted")
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--device")
    args = p.parse_args()
    phonemize = Phonemizer(args.checkpoint, args.device, args.beam)
    lines = [" ".join(args.text)] if args.text else [line.rstrip("\n") for line in sys.stdin]
    for start in range(0, len(lines), 64):
        for line in phonemize(lines[start:start + 64]):
            print(line)
