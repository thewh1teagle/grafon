# Evaluation

The `eval` TSV in each language YAML is what training scores every 10 minutes. It comes from the same teacher as the training data, so it measures imitation. The benchmarks below are independent, human-labelled references and say how good a model actually is.

## Hebrew: MILIM-Bench

[renikud/MILIM-Bench](https://huggingface.co/datasets/renikud/MILIM-Bench): 3,110 target words in context, in categories such as homographs, stress minimal pairs, names, acronyms, slang, gender and foreign words. `data/gold.tsv` has `Category<TAB>Text<TAB>Label`, where the label names the target word by position (`3=ʔitˈaχ`).

```bash
uv run python scripts/eval_milim.py runs/he-base/last
```

It scores per target word, both sides reduced to phoneme characters with stress kept, and prints accuracy per category.

| model | overall |
|---|---:|
| `he-base` step 7,399 | 81.8% |
| heb-g2pw (baseline) | 83.7% |
| neo2c (teacher) | 86.3% |

## Slovak: neurlang multi_eval

[neurlang/dataset `slovak/multi_eval.tsv`](https://github.com/neurlang/dataset/blob/master/slovak/multi_eval.tsv): 1,214 lowercase sentences with 1,315 target words. The second column has one token per word, `_` for words that are not scored:

```text
tiché pokojné a harmonické miesto v obklopení ...	_ _ _ _ _ _ ɔbklɔpɛɲiː _ ...
```

Its symbols differ from ours, so score after mapping both sides to one set: `ɛ` → `e`, `ɔ` → `o`, `ts` → `ʦ`, `tʃ` → `ʧ`, `dz` → `ʣ`, `dʒ` → `ʤ`, `g` → `ɡ`, and drop stress (`ˈ`), which it does not mark. No scoring script yet.

## Arabic: Arabic Speech Corpus

[halabi2016/arabic_speech_corpus](https://huggingface.co/datasets/halabi2016/arabic_speech_corpus) ([paper](https://aclanthology.org/L16-1116/)): 1,813 Modern Standard Arabic sentences with hand-checked phoneme transcriptions, stress included, aligned to one speaker's audio. It is the only human-verified MSA phoneme set at sentence level.

Its phonemes are Buckwalter-style, not IPA, so scoring maps them to our symbols first. No scoring script yet. The license is CC BY-NC-SA: use it to evaluate, never to train.

The training labels come from a different, unrelated teacher: [CATT](https://github.com/abjadai/catt) restores the vowels and our own rules turn vowelled text into IPA. The [CATT benchmark](https://github.com/abjadai/catt) (742 hand-diacritized news sentences) measures that diacritizer, not phonemes.
