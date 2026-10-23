"""Aligned contextual word examples for Urdu -> IPA.

Each Urdu space-delimited word produces exactly one IPA target. Context
never includes reference IPA, and the marked target is explicit.
"""
from __future__ import annotations

from .text import normalize

LEFT, RIGHT = "⟦", "⟧"


def context_source(words: list[str], index: int, radius: int = 2) -> str:
    if not words or not 0 <= index < len(words):
        raise ValueError("Invalid target word index")
    if radius < 0:
        raise ValueError("Context radius must be nonnegative")
    lo, hi = max(0, index - radius), min(len(words), index + radius + 1)
    neighbors = words[lo:hi].copy()
    if any(LEFT in w or RIGHT in w for w in neighbors):
        raise ValueError("Reserved target delimiters already occur in source text")
    neighbors[index-lo] = LEFT + neighbors[index-lo] + RIGHT
    return " ".join(neighbors)


def expand_rows(rows: list[tuple[str, str]], radius: int = 2) -> list[tuple[str, str]]:
    examples = []
    for line, (text, ipa) in enumerate(rows, 1):
        words, phones = normalize(text).split(), normalize(ipa).split()
        if len(words) != len(phones):
            raise ValueError(f"Row {line}: unaligned Urdu/IPA token counts")
        for i, gold_word in enumerate(phones):
            examples.append((context_source(words, i, radius), gold_word))
    return examples
