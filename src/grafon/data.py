"""TSV `text<TAB>phonemes` rows → encoder inputs, character ids and per-word phoneme targets.

A language is two inventories and a list of named normalizers. A word is a maximal run of graphemes on the text side and
of phonemes on the target side. Rows pair by whitespace token, and within a token by run;
a token whose run counts differ has no target, and a row whose token counts differ has
none at all. Unpaired words still reach the encoder and the character stack as context.
"""
from __future__ import annotations

import glob
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import regex as re
import torch
import yaml
from torch.utils.data import Dataset

IGNORE = -100
PAD, OTHER, SPACE = 0, 1, 2  # character ids; graphemes follow
BOS, EOS = 1, 2  # phoneme ids; 0 is padding, phonemes follow

MARKS = re.compile(r"\p{M}")
# Applied in order between NFC passes; the result is what the encoder, the character stack and the output see.
NORMALIZERS = {
    "strip_marks": lambda text: unicodedata.normalize("NFC", MARKS.sub("", unicodedata.normalize("NFD", text))),
    "lowercase": str.lower,
}


@dataclass(frozen=True)
class Language:
    graphemes: str
    phonemes: str
    normalizers: tuple = ()

    @classmethod
    def from_dict(cls, settings):
        settings = dict(settings)
        strip = settings.pop("strip", None)  # checkpoints before normalizers stored a mark-stripping regex
        if strip:
            if strip != r"\p{M}":
                raise ValueError(f"legacy strip {strip!r} has no normalizer")
            settings["normalizers"] = ["strip_marks"]
        return cls(**settings)

    def __post_init__(self):
        for name in ("graphemes", "phonemes"):
            value = getattr(self, name)
            if len(set(value)) != len(value) or any(c.isspace() for c in value):
                raise ValueError(f"{name} must be distinct, non-whitespace characters")
        object.__setattr__(self, "word_re", re.compile(f"[{re.escape(self.graphemes)}]+"))
        object.__setattr__(self, "phoneme_re", re.compile(f"[{re.escape(self.phonemes)}]+"))
        object.__setattr__(self, "normalizers", tuple(self.normalizers))
        unknown = [n for n in self.normalizers if n not in NORMALIZERS]
        if unknown:
            raise ValueError(f"unknown normalizers {unknown}; known: {sorted(NORMALIZERS)}")
        object.__setattr__(self, "char_index", {c: i + 3 for i, c in enumerate(self.graphemes)})
        object.__setattr__(self, "phoneme_index", {c: i + 3 for i, c in enumerate(self.phonemes)})

    def to_dict(self):
        return dict(graphemes=self.graphemes, phonemes=self.phonemes, normalizers=list(self.normalizers))

    @property
    def char_vocab(self):
        return len(self.graphemes) + 3

    @property
    def phoneme_vocab(self):
        return len(self.phonemes) + 3

    def normalize(self, text):
        text = unicodedata.normalize("NFC", text)
        for name in self.normalizers:
            text = NORMALIZERS[name](text)
        return unicodedata.normalize("NFC", text)

    def char_id(self, char):
        return self.char_index.get(char, SPACE if char.isspace() else OTHER)

    def words(self, text):
        return [m.span() for m in self.word_re.finditer(text)]

    def decode(self, ids):
        return "".join(self.phonemes[i - 3] for i in ids if i >= 3)

    def pair(self, text, phonemes):
        """Word spans of `text` and, per span, its phoneme ids or None when unpaired."""
        spans, targets = [], []
        tokens = list(re.finditer(r"\S+", text))
        refs = phonemes.split()
        for k, token in enumerate(tokens):
            runs = [(token.start() + a, token.start() + b) for a, b in self.words(token.group())]
            said = self.phoneme_re.findall(refs[k]) if len(tokens) == len(refs) else []
            # A letter or mark outside the inventory masks the token; punctuation and digits pass.
            clean = len(tokens) == len(refs) and not any(
                unicodedata.category(c)[0] in "LM" and c not in self.phoneme_index for c in refs[k])
            paired = clean and len(said) == len(runs)
            spans += runs
            targets += [[self.phoneme_index[c] for c in s] for s in said] if paired else [None] * len(runs)
        return spans, targets


@dataclass(frozen=True)
class Config:
    """A language YAML: the Language plus the run's backbone, training TSVs and eval TSV."""
    language: Language
    backbone: str
    train: tuple
    eval: str

    @classmethod
    def load(cls, path):
        settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        run = {key: settings.pop(key, None) for key in ("backbone", "train", "eval")}
        missing = [key for key, value in run.items() if value is None]
        if missing:
            raise ValueError(f"{path} must set {', '.join(missing)}")
        patterns = [run["train"]] if isinstance(run["train"], str) else run["train"]
        train = tuple(sorted(f for pattern in patterns for f in glob.glob(pattern)))
        if not train or not Path(run["eval"]).is_file():
            raise ValueError(f"{path}: training or eval TSV not found")
        return cls(Language.from_dict(settings), run["backbone"], train, run["eval"])


def line_index(path):
    """Byte offsets of every non-empty line, built with one vectorized scan."""
    data = np.memmap(path, dtype=np.uint8, mode="r")
    ends = np.flatnonzero(data == ord("\n"))
    if len(data) and data[-1] != ord("\n"):
        ends = np.append(ends, len(data))
    starts = np.concatenate([[0], ends[:-1] + 1])
    keep = ends > starts
    return starts[keep].astype(np.int64), (ends - starts)[keep].astype(np.int64)


class TSV(Dataset):
    """Rows of one or more TSV files, read lazily through a byte-offset index."""

    def __init__(self, paths):
        self.paths = [str(p) for p in paths]
        self.index = []
        for number, path in enumerate(self.paths):
            starts, lengths = line_index(path)
            self.index.append(np.stack([np.full_like(starts, number), starts, lengths], 1))
        self.index = np.concatenate(self.index)
        self.handles = None

    def __len__(self):
        return len(self.index)

    def lengths(self):
        return self.index[:, 2]

    def __getitem__(self, i):
        if self.handles is None:
            self.handles = [open(p, "rb") for p in self.paths]
        number, start, length = self.index[i]
        handle = self.handles[number]
        handle.seek(start)
        text, _, phonemes = handle.read(length).decode("utf-8").rstrip("\r").partition("\t")
        return text, phonemes

    def __getstate__(self):
        return dict(self.__dict__, handles=None)


class Batches:
    """Seeded length-bucketed batches under a character budget, sharded across ranks.

    Epoch e's batches depend only on (seed, e), so a resume skips exactly the batches
    already consumed and a run's training rows are reproducible from its seed and step.
    """

    def __init__(self, lengths, budget, seed, rank=0, world=1, shuffle=True, pool=256):
        self.lengths, self.budget, self.seed = lengths, budget, seed
        self.rank, self.world, self.shuffle, self.pool = rank, world, shuffle, pool
        self.epoch, self.skip, self._cache = 0, 0, None

    def batches(self, epoch):
        rng = random.Random(self.seed * 1000003 + epoch)
        order = np.arange(len(self.lengths))
        if self.shuffle:
            order = np.random.default_rng(rng.randrange(2**63)).permutation(order)
        batches = []
        for group in np.array_split(order, max(1, len(order) // (self.pool * 64))):
            group = group[np.argsort(self.lengths[group], kind="stable")]
            batch, longest = [], 0
            for i in group:
                longest = max(longest, int(self.lengths[i]))
                if batch and longest * (len(batch) + 1) > self.budget:
                    batches.append(batch)
                    batch, longest = [], int(self.lengths[i])
                batch.append(int(i))
            if batch:
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        # Every rank takes the same number of steps.
        return batches[:len(batches) // self.world * self.world][self.rank::self.world]

    def epoch_batches(self):
        if self._cache is None or self._cache[0] != self.epoch:
            self._cache = (self.epoch, self.batches(self.epoch))
        return self._cache[1]

    def __len__(self):
        return len(self.epoch_batches())

    def __iter__(self):
        batches, skip = self.epoch_batches(), self.skip
        self.skip = 0
        yield from batches[skip:]


class Collate:
    def __init__(self, language, tokenizer, max_chars=1024, max_tokens=512, max_word_chars=64, max_phonemes=64):
        self.language, self.tokenizer = language, tokenizer
        self.max_chars, self.max_tokens = max_chars, max_tokens
        self.max_word_chars, self.max_phonemes = max_word_chars, max_phonemes

    def rows(self, pairs):
        """Normalize and pair rows; phonemes may be None for inference."""
        rows = []
        for text, phonemes in pairs:
            text = self.language.normalize(text)
            if phonemes is None:
                spans, targets = self.language.words(text), None
            else:
                phonemes = unicodedata.normalize("NFC", phonemes)
                spans, targets = self.language.pair(text, phonemes)
            if len(text) > self.max_chars:
                if phonemes is None:
                    raise ValueError(f"{len(text)} characters exceed max_chars {self.max_chars}; split the text")
                # Training rows keep their first max_chars characters and the words inside them.
                keep = [k for k, (_, end) in enumerate(spans) if end <= self.max_chars]
                text, spans = text[:self.max_chars], [spans[k] for k in keep]
                targets = [targets[k] for k in keep]
            rows.append(dict(text=text, phonemes=phonemes, spans=spans, targets=targets))
        return rows

    def __call__(self, pairs):
        return self.encode(self.rows(pairs))

    def encode(self, rows):
        texts = [row["text"] for row in rows]
        chars = max(1, max(len(t) for t in texts))
        char_ids = torch.zeros(len(rows), chars, dtype=torch.long)
        char_to_token = torch.full((len(rows), chars), -1, dtype=torch.long)
        within = torch.zeros(len(rows), chars, dtype=torch.long)
        if self.tokenizer is not None:
            encoded = self.tokenizer(texts, return_offsets_mapping=True, truncation=True, max_length=self.max_tokens,
                                     padding=True, return_tensors="pt", return_token_type_ids=False)
            offsets = encoded.pop("offset_mapping").tolist()
        words, targets = [], []
        for b, row in enumerate(rows):
            text = row["text"]
            char_ids[b, :len(text)] = torch.tensor([self.language.char_id(c) for c in text], dtype=torch.long)
            if self.tokenizer is not None:
                for token, (start, end) in enumerate(offsets[b]):
                    char_to_token[b, start:end] = token
                    within[b, start:end] = torch.arange(end - start)
            for k, (start, end) in enumerate(row["spans"]):
                target = row["targets"][k] if row["targets"] is not None else None
                # A word past the encoder window or over the length caps is not supervised.
                if target is not None and (end - start > self.max_word_chars or len(target) + 1 > self.max_phonemes
                                           or (self.tokenizer is not None and char_to_token[b, end - 1] < 0)):
                    target = None
                words.append((b, start, min(end, start + self.max_word_chars)))
                targets.append(target)
        batch = dict(char_ids=char_ids, char_to_token=char_to_token, within=within, rows=rows)
        if self.tokenizer is not None:
            batch.update(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        width = max([end - start for _, start, end in words], default=1)
        index = torch.full((len(words), width), -1, dtype=torch.long)
        for w, (b, start, end) in enumerate(words):
            index[w, :end - start] = b * chars + torch.arange(start, end)
        supervised = [w for w, t in enumerate(targets) if t is not None]
        length = max([len(targets[w]) + 1 for w in supervised], default=1)
        decoder_input = torch.zeros(len(supervised), length, dtype=torch.long)
        labels = torch.full((len(supervised), length), IGNORE, dtype=torch.long)
        for i, w in enumerate(supervised):
            ids = targets[w]
            decoder_input[i, :len(ids) + 1] = torch.tensor([BOS] + ids)
            labels[i, :len(ids) + 1] = torch.tensor(ids + [EOS])
        batch.update(word_chars=index, supervised=torch.tensor(supervised, dtype=torch.long),
                     decoder_input=decoder_input, labels=labels, word_rows=[b for b, _, _ in words],
                     word_targets=targets)
        return batch


def assemble(text, spans, predictions):
    """Replace each word span with its phonemes; everything else passes through."""
    out, last = [], 0
    for (start, end), phonemes in zip(spans, predictions):
        out += [text[last:start], phonemes]
        last = end
    return "".join(out + [text[last:]])
