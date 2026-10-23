"""Compact Urdu G2P contribution for Grafon."""
__all__ = ["Phonemizer"]

def __getattr__(name):
    if name == "Phonemizer":
        from .infer import Phonemizer
        return Phonemizer
    raise AttributeError(name)
