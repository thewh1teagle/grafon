"""grafon: text → phonemes with any encoder; everything outside the grapheme inventory passes through."""
from .infer import Phonemizer, main

__all__ = ["Phonemizer", "main"]
