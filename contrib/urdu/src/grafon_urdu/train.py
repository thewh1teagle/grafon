"""Reproducible compact Urdu G2P training. Train split only builds vocabulary."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

from .model import CompactG2P, ModelConfig
from .text import read_tsv, vocabulary, encode, decode, save_json, sha256, graphemes
from .context import expand_rows


class Pairs(Dataset):
    def __init__(self, rows, src_vocab, tgt_vocab, max_src, max_tgt):
        self.examples = []
        rejected = []
        for idx, (src, tgt) in enumerate(rows, 1):
            a, b = encode(src, src_vocab), encode(tgt, tgt_vocab, bos=True)
            if len(a) > max_src or len(b) > max_tgt:
                rejected.append((idx, len(a), len(b)))
                continue
            self.examples.append((a, b))
        if rejected:
            sample = rejected[:3]
            raise ValueError(f"{len(rejected)} samples exceed length limits. Sample (line,src,tgt): {sample}. "
                             "Adjust --max-source/--max-target; refusing to silently drop data.")

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


def collate(batch):
    batch_size = len(batch)
    src = torch.zeros((batch_size, max(len(a) for a, _ in batch)), dtype=torch.long)
    target = torch.zeros((batch_size, max(len(b) for _, b in batch)), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        src[i, :len(a)] = torch.tensor(a)
        target[i, :len(b)] = torch.tensor(b)
    return src, target


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def cross_entropy(logits, target):
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=0
    )


@torch.no_grad()
def val_loss(model, loader, device, max_batches):
    model.eval()
    weighted_sum, tokens, batches = 0.0, 0, 0
    for src, full_tgt in loader:
        if max_batches and batches >= max_batches: break
        src, full_tgt = src.to(device), full_tgt.to(device)
        output = full_tgt[:, 1:]
        loss = cross_entropy(model(src, full_tgt[:, :-1]), output)
        count = int(output.ne(0).sum().item())
        weighted_sum += float(loss.item()) * count
        tokens += count
        batches += 1
    return weighted_sum / max(1, tokens)


def main():
    ap = argparse.ArgumentParser(description="Train a compact Urdu character-to-IPA encoder decoder")
    ap.add_argument("--train", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--training-unit", choices=["sentence", "word_context"], default="sentence")
    ap.add_argument("--context-radius", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--max-source", type=int, default=384)
    ap.add_argument("--max-target", type=int, default=640)
    ap.add_argument("--max-train-rows", type=int, default=0, help="Smoke-test only")
    ap.add_argument("--max-val-batches", type=int, default=32, help="0 for full validation")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1: ap.error("epochs and batch-size must be positive")
    if args.context_radius < 0: ap.error("context-radius must be nonnegative")
    if args.d_model % args.heads != 0: ap.error("d-model must be divisible by heads")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_rows_all = read_tsv(args.train)
    valid_rows = read_tsv(args.validation)
    if set(text for text, _ in train_rows_all) & set(text for text, _ in valid_rows):
        raise ValueError("Exact train/validation sentence overlap detected")
    # Select sentence rows first, then create contextual word examples.
    # The split-specific word contexts never cross a sentence or data split.
    train_rows = train_rows_all[:args.max_train_rows] if args.max_train_rows else train_rows_all
    if args.training_unit == "word_context":
        train_rows = expand_rows(train_rows, radius=args.context_radius)
        valid_rows = expand_rows(valid_rows, radius=args.context_radius)
        # Vocabulary is built from the COMPLETE training split only,
        # including held-out source markers, never validation/test data.
        vocab_train_rows = expand_rows(train_rows_all, radius=args.context_radius)
    else:
        vocab_train_rows = train_rows_all
    src_vocab = vocabulary([x for x, _ in vocab_train_rows])
    tgt_vocab = vocabulary([y for _, y in vocab_train_rows])
    del vocab_train_rows
    train_ds = Pairs(train_rows, src_vocab, tgt_vocab, args.max_source, args.max_target)
    val_ds = Pairs(valid_rows, src_vocab, tgt_vocab, args.max_source, args.max_target)
    cfg = ModelConfig(len(src_vocab), len(tgt_vocab), args.d_model, args.heads,
                      args.layers, args.layers, args.ffn)
    model = CompactG2P(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    autocast_enabled = bool(args.bf16 and str(args.device).startswith("cuda"))
    if autocast_enabled and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA GPU does not support bf16")
    print(json.dumps({"training_unit": args.training_unit, "context_radius": args.context_radius,
                      "train_rows": len(train_ds), "validation_rows": len(val_ds),
                      "parameter_count": model.parameters_count(), "checkpoint_size_fp32_MB":
                      round(model.parameters_count() * 4 / 1e6, 2),
                      "device": args.device, "bf16": autocast_enabled,
                      "training_sha256": sha256(args.train),
                      "validation_sha256": sha256(args.validation)}, ensure_ascii=False), flush=True)
    best = float("inf")
    total_steps, global_step = args.epochs * len(train_loader), 0
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        for src, target in train_loader:
            src, target = src.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            # Use torch autocast context for bf16 compute and fp32 parameter storage.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = model(src, target[:, :-1])
                loss = cross_entropy(logits, target[:, 1:])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            epoch_loss += float(loss.item())
            if global_step % args.log_every == 0:
                print(f"epoch={epoch} step={global_step}/{total_steps} loss={epoch_loss / (global_step - (epoch-1)*len(train_loader)):.5f}", flush=True)
        # Full precision validation avoids autocast-induced comparison drift.
        validation_loss = val_loss(model, val_loader, args.device, args.max_val_batches)
        stats = {"epoch": epoch, "steps": global_step, "train_loss": epoch_loss / len(train_loader),
                 "validation_loss": validation_loss, "validation_batches":
                 min(len(val_loader), args.max_val_batches) if args.max_val_batches else len(val_loader),
                 "validation_reference": "reviewed/mixed-provenance; not independent gold"}
        print("EVAL " + json.dumps(stats), flush=True)
        if validation_loss < best:
            best = validation_loss
            best_dir = out / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                      str(best_dir / "model.safetensors"))
            save_json(best_dir / "config.json", vars(cfg))
            save_json(best_dir / "preprocessing.json", {"training_unit": args.training_unit, "context_radius": args.context_radius})
            save_json(best_dir / "source_vocab.json", src_vocab)
            save_json(best_dir / "target_vocab.json", tgt_vocab)
            save_json(best_dir / "training_report.json", {**stats, "parameter_count": model.parameters_count(),
                      "train_sha256": sha256(args.train), "validation_sha256": sha256(args.validation),
                      "selection_metric": "validation_loss", "training_unit": args.training_unit,
                      "context_radius": args.context_radius})
        save_json(out / "last_epoch.json", stats)
    print("Best checkpoint:", out / "best", flush=True)


if __name__ == "__main__":
    main()
