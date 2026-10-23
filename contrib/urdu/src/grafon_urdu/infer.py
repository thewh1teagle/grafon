"""Local and Hugging Face Hub model loading / command line inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .model import CompactG2P, ModelConfig
from .text import decode, encode
from .context import context_source


class Phonemizer:
    def __init__(self, model: CompactG2P, src_vocab: dict, tgt_vocab: dict, device: str,
                 training_unit: str = "sentence", context_radius: int = 2):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.training_unit = training_unit
        self.context_radius = context_radius

    @classmethod
    def from_pretrained(cls, path_or_repo: str, device: str = "cpu") -> "Phonemizer":
        path = Path(path_or_repo)
        if not path.is_dir():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError("Install grafon-urdu-g2p[hub] to fetch from Hugging Face") from exc
            path = Path(snapshot_download(repo_id=path_or_repo, repo_type="model", allow_patterns=[
                "model.safetensors", "config.json", "source_vocab.json", "target_vocab.json", "preprocessing.json"
            ]))
        cfg = ModelConfig(**json.loads((path / "config.json").read_text(encoding="utf-8")))
        src_vocab = json.loads((path / "source_vocab.json").read_text(encoding="utf-8"))
        tgt_vocab = json.loads((path / "target_vocab.json").read_text(encoding="utf-8"))
        model = CompactG2P(cfg)
        model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
        model = model.to(device).eval()
        settings_path = path / "preprocessing.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
        return cls(model, src_vocab, tgt_vocab, device,
                   settings.get("training_unit", "sentence"), settings.get("context_radius", 2))

    @torch.inference_mode()
    def phonemize_many(self, sentences: list[str], batch_size: int = 48,
                       max_new_tokens: int = 96) -> list[str]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.training_unit != "word_context":
            return [self._phonemize_sentence(s, max_new_tokens=480) for s in sentences]
        from .text import normalize
        from torch.nn.utils.rnn import pad_sequence
        contexts, counts = [], []
        for text in sentences:
            words = normalize(text).split()
            counts.append(len(words))
            for i in range(len(words)):
                contexts.append(context_source(words, i, self.context_radius))
        ipa_words = []
        for start in range(0, len(contexts), batch_size):
            enc = [encode(x, self.src_vocab) for x in contexts[start:start+batch_size]]
            if any(len(x) > 384 for x in enc):
                raise ValueError("Context source exceeds 384 grapheme tokens")
            tensor = pad_sequence([torch.tensor(x, dtype=torch.long) for x in enc],
                                  batch_first=True, padding_value=0).to(self.device)
            decoded = self.model.generate(tensor, max_new_tokens=max_new_tokens)
            for ids in decoded:
                if 3 not in ids:
                    raise RuntimeError("Word decoding reached max_new_tokens without EOS; do not silently truncate")
                word = decode(ids, self.tgt_vocab)
                if not word or len(word.split()) != 1:
                    raise RuntimeError(f"Word decoder produced invalid IPA unit: {word!r}")
                ipa_words.append(word)
        results = []
        pos = 0
        for n in counts:
            results.append(" ".join(ipa_words[pos:pos+n]))
            pos += n
        assert pos == len(ipa_words)
        return results

    def _phonemize_sentence(self, text: str, max_new_tokens: int) -> str:
        src = torch.tensor([encode(text, self.src_vocab)], dtype=torch.long, device=self.device)
        if src.shape[1] > 384:
            raise ValueError("Input too long (>383 source grapheme tokens); split into shorter sentences")
        ids = self.model.generate(src, max_new_tokens=max_new_tokens)[0]
        return decode(ids, self.tgt_vocab)

    @torch.inference_mode()
    def phonemize(self, text: str, max_new_tokens: int = 480) -> str:
        if self.training_unit == "word_context":
            return self.phonemize_many([text])[0]
        return self._phonemize_sentence(text, max_new_tokens)

    def __call__(self, text: str) -> str:
        return self.phonemize(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact Urdu grapheme-to-IPA inference")
    parser.add_argument("--model", required=True, help="Local model dir or HF repo id")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("text", nargs="?")
    parser.add_argument("--file", help="UTF-8 file containing one sentence per line")
    args = parser.parse_args()
    if not args.file and not args.text:
        parser.error("Pass text or --file")
    ph = Phonemizer.from_pretrained(args.model, device=args.device)
    sentences = Path(args.file).read_text(encoding="utf-8").splitlines() if args.file else [args.text]
    for sentence in sentences:
        print(ph(sentence))


if __name__ == "__main__":
    main()
