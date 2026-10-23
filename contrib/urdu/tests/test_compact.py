from pathlib import Path
import pytest
import torch
from grafon_urdu.text import read_tsv, graphemes, vocabulary, encode, decode
from grafon_urdu.model import CompactG2P, ModelConfig
from grafon_urdu.train import Pairs, collate
from grafon_urdu.eval import distance, evaluate


def test_grapheme_combining():
    assert len(graphemes("t̪")) == 1
    assert len(graphemes("mɛ̃")) == 2


def test_vocab_roundtrip():
    words = ["مجھے اردو", "کراچی"]
    vocab = vocabulary(words)
    assert decode(encode(words[0], vocab), vocab) == words[0]


def test_alignment_reject(tmp_path):
    p = tmp_path / "a.tsv"
    p.write_text("آج ہے۔\taːdʒ\n", encoding="utf-8")
    with pytest.raises(ValueError): read_tsv(p)
    p.write_text("آج\t[MASK]\n", encoding="utf-8")
    with pytest.raises(ValueError): read_tsv(p)


def test_model_forward():
    cfg = ModelConfig(16, 16, d_model=32, nhead=4, encoder_layers=1,
                      decoder_layers=1, feedforward_dim=64, dropout=0)
    model = CompactG2P(cfg)
    src = torch.tensor([[4,5,3,0],[6,3,0,0]])
    tgt = torch.tensor([[2,7,8],[2,4,0]])
    y = model(src, tgt)
    assert y.shape == (2,3,16)
    assert model.parameters_count() < 100_000


def test_model_generate():
    model = CompactG2P(ModelConfig(12,12,32,4,1,1,64,0))
    assert len(model.generate(torch.tensor([[4,5,3]]), max_new_tokens=3)[0]) <= 3


def test_dataset_length_limit():
    src = vocabulary(["ایک جملہ"])
    tgt = vocabulary(["eːk dʒʊmla"])
    with pytest.raises(ValueError): Pairs([("ایک جملہ", "eːk dʒʊmla")], src, tgt, 3, 5)


def test_distance():
    assert distance(graphemes("aː"), graphemes("a")) == 1
    assert distance([], []) == 0


def test_checkpoint_roundtrip(tmp_path):
    from safetensors.torch import save_file
    from grafon_urdu.text import save_json
    from grafon_urdu.infer import Phonemizer
    cfg=ModelConfig(10,10,32,4,1,1,64,0)
    model=CompactG2P(cfg)
    save_file({k:v.cpu().contiguous() for k,v in model.state_dict().items()},str(tmp_path/'model.safetensors'))
    save_json(tmp_path/'config.json',vars(cfg))
    src={"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3,"ا":4,"ب":5}
    tgt={"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3,"a":4,"b":5}
    save_json(tmp_path/'source_vocab.json',src)
    save_json(tmp_path/'target_vocab.json',tgt)
    ph=Phonemizer.from_pretrained(str(tmp_path),device='cpu')
    assert isinstance(ph.phonemize('اب',max_new_tokens=3),str)
