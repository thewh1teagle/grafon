"""Unicode grapheme tokenizer and aligned TSV dataset reader."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
import regex

SPECIAL = ("<pad>", "<unk>", "<bos>", "<eos>")


def graphemes(text: str) -> list[str]:
    return regex.findall(r"\X", unicodedata.normalize("NFC", text))


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def read_tsv(path: str | Path) -> list[tuple[str, str]]:
    data = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"{path}:{number}: expected exactly two tab-separated columns")
        text, ipa = map(normalize, parts)
        if not text or not ipa or "[MASK]" in ipa:
            raise ValueError(f"{path}:{number}: empty text/IPA or masked label")
        if len(text.split()) != len(ipa.split()):
            raise ValueError(f"{path}:{number}: Urdu/IPA word boundaries not aligned")
        data.append((text, ipa))
    if not data:
        raise ValueError(f"{path}: no rows")
    return data


def vocabulary(texts: list[str]) -> dict[str, int]:
    counts = Counter(char for s in texts for char in graphemes(s))
    result = {token: i for i, token in enumerate(SPECIAL)}
    for char, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if char not in result:
            result[char] = len(result)
    return result


def encode(text: str, vocab: dict[str, int], *, bos: bool = False) -> list[int]:
    tokens = ([vocab["<bos>"]] if bos else []) + [
        vocab.get(g, vocab["<unk>"]) for g in graphemes(text)
    ] + [vocab["<eos>"]]
    return tokens


def decode(ids: list[int], vocab: dict[str, int]) -> str:
    lookup = {v: k for k, v in vocab.items()}
    seq = []
    for i in ids:
        token = lookup.get(int(i), "<unk>")
        if token == "<eos>":
            break
        if token not in SPECIAL:
            seq.append(token)
    return "".join(seq)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
