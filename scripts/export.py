"""Training checkpoint → a Hub-ready folder: config, fp16 weights, tokenizer and a model card.

uv run scripts/export.py runs/sk-base/best exports/grafon-sk
uv run scripts/export.py runs/sk-base/best exports/grafon-sk --push grafon-g2p/sk
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TRAINING_ONLY = {"model.safetensors", "optimizer.bin", "scheduler.bin", "trainer.json"}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

CARD = """---
library_name: grafon
language: {language}
base_model: {backbone}
pipeline_tag: text-to-speech
tags: [g2p, phonemes, ipa, grapheme-to-phoneme]
---

# {name}

Contextual grapheme-to-phoneme for {language_name}, trained with [grafon](https://github.com/thewh1teagle/grafon) on top of [{backbone}](https://huggingface.co/{backbone}).

```python
from grafon import Phonemizer

phonemize = Phonemizer.from_pretrained("{repo}")
print(phonemize({example!r}))
```

Phonemes: `{phonemes}`. Anything outside the letters `{graphemes}` passes through as written.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--dtype", choices=DTYPES, default="fp16", help="Stored weight precision; loads into fp32")
    p.add_argument("--language", default="sk", help="ISO code for the model card")
    p.add_argument("--language-name", default="Slovak")
    p.add_argument("--example", default="Ahoj svet, čo si myslíš o modeli?")
    p.add_argument("--push", metavar="REPO", help="Upload the folder to this Hub repo (created public if missing)")
    args = p.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for f in args.checkpoint.iterdir():
        if f.is_file() and f.name not in TRAINING_ONLY and not f.name.startswith("random_states"):
            shutil.copy2(f, args.out / f.name)
    weights = load_file(str(args.checkpoint / "model.safetensors"))
    dtype = DTYPES[args.dtype]
    save_file({k: v.to(dtype) if v.is_floating_point() else v for k, v in weights.items()},
              str(args.out / "model.safetensors"))

    config = json.loads((args.checkpoint / "config.json").read_text())
    repo = args.push or f"<user>/{args.out.name}"
    (args.out / "README.md").write_text(CARD.format(
        name=repo.split("/")[-1], repo=repo, backbone=config["backbone"], language=args.language,
        language_name=args.language_name, example=args.example,
        phonemes=config["language"]["phonemes"], graphemes=config["language"]["graphemes"]), encoding="utf-8")
    size = sum(f.stat().st_size for f in args.out.iterdir()) / 1e6
    print(f"{args.out}: {', '.join(sorted(f.name for f in args.out.iterdir()))} ({size:.0f} MB)")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, exist_ok=True)
        api.upload_folder(repo_id=args.push, folder_path=str(args.out))
        print(f"https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
