# Architecture

**Any text encoder → contextual character states → a small autoregressive decoder per word.**

```text
text ─┬─ encoder subwords (any HF model with a fast tokenizer) ─┐
      │                                                          ├─ character stack (whole sentence) ─ word decoder ─ phonemes
      └─ characters ─────────────────────────────────────────────┘
everything outside the grapheme inventory ─────────────────────────────────── passthrough
```

## Language

A language is two inventories and a list of named normalizers, in YAML (`configs/he.yaml`, `configs/sk.yaml`):

```yaml
graphemes: אבגדהוזחטיכךלמםנןסעפףצץקרשת׳״'"
phonemes: abdefhijklmnopstuvwzɡʁʃʒʔˈχ
normalizers: [strip_marks]
```

- A word is a maximal run of graphemes. Everything else (digits, Latin, punctuation) passes through untouched but still reaches the encoder as context.
- Character ids: pad, other, space, then the graphemes. Phoneme ids: pad, bos, eos, then the phonemes. Stress (`ˈ`) is an ordinary phoneme.
- Normalizers run in order between NFC passes: `nfkc` (folds presentation forms and ligatures), `strip_marks` (removes the combining marks left after NFC, such as niqqud and harakat; precomposed letters like أ keep theirs) and `lowercase`. Their output is what the encoder, the character stack and the passthrough all see. Checkpoints with the older `strip: '\p{M}'` load as `[strip_marks]`.
- The inventory is saved in each checkpoint's `config.json`; the YAML is read only when a run starts.
- The YAML also sets the run's `backbone`, `train` and `eval`, so a run is `--language configs/sk.yaml --output runs/sk-base`. They are not part of the `Language`.

## Data

TSV `text<TAB>phonemes`. Words pair by whitespace token, then by run inside a token (grapheme runs on one side, phoneme runs on the other). A token whose run counts differ has no target; a row whose token counts differ has none at all. Unpaired words still reach the encoder and the character stack; they only contribute no loss.

Batches are length-bucketed under a byte budget and seeded by `(seed, epoch)`, so a resume skips exactly the consumed batches.

## Model

- **Encoder contract:** a fast tokenizer with `offset_mapping`. Offsets map each character to its subword; nothing else about the encoder is assumed. `context_proj` maps `hidden_size` to the stack width, the one place the encoder's size appears. `--backbone none` trains on characters alone.
- **Character stack:** character embedding + gathered subword context + within-subword position + sentence position, then pre-norm transformer layers over the whole sentence. Characters outside the encoder window keep their letters and get no context.
- **Word decoder:** each word's character states (plus a within-word position) are gathered into one flat batch of all words in the batch. A pre-norm transformer decoder cross-attends to them and emits phonemes until EOS. Training is teacher-forced cross-entropy over paired words.
- **Decoding:** greedy or beam search with a KV cache; the word's cross-attention keys and values are computed once. Words decode in parallel and independently: coherence within a word comes from autoregression, across words only from the shared context in the character states.

## Inference

`Phonemizer(checkpoint)` or `uv run grafon <checkpoint> "text"`: normalize → word spans → model → each span replaced by its phonemes, everything else kept as written.

On aarch64, torch's oneDNN path is slow for the decoder's small matrices; `torch.backends.mkldnn.enabled = False` cut CPU decoding about 3.5×. The encoder dominates latency.

## Sizes

Width 768, 12 heads, 2 stack layers, 2 decoder layers are placeholders carried over from heb-g2pw, not measured. The decoder costs about as much time per training step as the 362M encoder forward pass; a capacity sweep (`scripts/sweep_capacity.py`, not written yet) should set them.
