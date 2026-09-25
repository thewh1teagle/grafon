"""Phonemize with pretrained grafon models; weights download from the Hub on first use.

uv run examples/phonemize.py
"""
from grafon import Phonemizer

MODELS = {
    "grafon-g2p/sk": ["Ahoj svet, čo si myslíš o modeli?", "Idem s mamou k otcovi v Prahe."],
    "grafon-g2p/he": ["אחרי שאתה אורז את המזוודה, תבוא לאכול אורז"],
}

for repo, sentences in MODELS.items():
    phonemize = Phonemizer.from_pretrained(repo)
    for sentence, phonemes in zip(sentences, phonemize(sentences)):
        print(f"{sentence}\n  {phonemes}")
