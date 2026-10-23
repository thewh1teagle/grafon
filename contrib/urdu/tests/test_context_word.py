import json
import torch
from grafon_urdu.context import context_source, expand_rows
from grafon_urdu.infer import Phonemizer
from grafon_urdu.model import CompactG2P, ModelConfig
from grafon_urdu.text import vocabulary


def test_source_word_alignment_and_punctuation():
    rows = [("عمر نے کتاب رکھی۔", "ʊmər neː kɪt̪aːb rəkʰiː")]
    samples = expand_rows(rows)
    assert len(samples) == 4
    assert samples[0] == ("⟦عمر⟧ نے کتاب", "ʊmər")
    assert samples[3] == ("نے کتاب ⟦رکھی۔⟧", "rəkʰiː")


def test_disambiguating_context():
    a = context_source("عمر نے کتاب لکھی".split(), 0)
    b = context_source("عمر کتنی ہے".split(), 0)
    assert a != b
    assert a.startswith("⟦عمر⟧") and b.startswith("⟦عمر⟧")


def test_markers_forbidden_in_source():
    import pytest
    with pytest.raises(ValueError): context_source(["⟦لفظ"], 0)


def test_phonemize_many_preserves_word_count(monkeypatch):
    src = vocabulary(["⟦عمر⟧ نے کتاب", "نے ⟦کتاب⟧"])
    tgt = vocabulary(["ʊmər", "neː", "kɪt̪aːb"])
    model = CompactG2P(ModelConfig(len(src), len(tgt), 32, 4, 1, 1, 64, 0.0))
    def fake_generate(x, **kwargs):
        token = tgt['n'] if 'n' in tgt else tgt['ʊ']
        return [[token, 3] for _ in range(x.shape[0])]
    monkeypatch.setattr(model, 'generate', fake_generate)
    ph = Phonemizer(model, src, tgt, 'cpu', 'word_context', 2)
    output = ph.phonemize_many(["عمر نے کتاب", "ایک جملہ"], batch_size=2)
    assert [len(x.split()) for x in output] == [3, 2]


def test_old_checkpoint_still_sentence_mode():
    src = vocabulary(["عمر ہے"]); tgt = vocabulary(["ʊmər hɛ"])
    model = CompactG2P(ModelConfig(len(src), len(tgt), 32, 4, 1, 1, 64, 0.0))
    ph = Phonemizer(model, src, tgt, 'cpu')
    assert ph.training_unit == 'sentence'
