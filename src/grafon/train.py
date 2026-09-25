"""uv run accelerate launch -m grafon.train --output runs/<name>

Checkpoints: <output>/last on every eval, <output>/best when eval loss improves.
Stop with Ctrl-C or SIGTERM: the run checkpoints before it exits, and --resume <output>/last
continues it exactly. TensorBoard logs go to <output>/tensorboard.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import time
from pathlib import Path

import jiwer
import pyinject
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .data import TSV, Batches, Collate, Config, assemble
from .model import G2P, locate

TENSORS = ("input_ids", "attention_mask", "char_ids", "char_to_token", "within", "word_chars",
           "supervised", "decoder_input", "labels")


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", default="configs/he.yaml", help="Inventories, backbone, train and eval TSVs")
    p.add_argument("--output", required=True, help="Run directory, e.g. runs/he-base")
    p.add_argument("--resume", help="Checkpoint directory, e.g. runs/he-base/last")
    p.add_argument("--init", help="Start from a trained model's weights (Hub repo or checkpoint), fresh optimizer; "
                                  "sizes and backbone come from it")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-bytes", type=int, default=24000, help="Padded row bytes per GPU pass")
    p.add_argument("--lr", type=float, default=3e-5, help="Encoder learning rate")
    p.add_argument("--head-lr", type=float, default=3e-4, help="Character stack and decoder learning rate")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--eval-minutes", type=float, default=10)
    p.add_argument("--log-steps", type=int, default=50)
    p.add_argument("--beam", type=int, default=1, help="Beam width for eval decoding")
    # Placeholder sizes until scripts/sweep_capacity.py measures them.
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-chars", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.resume and args.init:
        p.error("--resume continues a run; --init starts a new one from trained weights; pick one")
    try:
        args.config = Config.load(args.language)
    except ValueError as error:
        p.error(str(error))
    return args


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if k in TENSORS and k in batch else v for k, v in batch.items()}


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


@torch.no_grad()
def evaluate(model, loader, device, beam):
    """Teacher-forced loss, per-word error on paired words, and full-sentence WER with passthrough."""
    model.eval()
    loss_sum = tokens = words = wrong = char_errors = chars = unpaired = 0
    references, hypotheses = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch)
            predicted = model.generate(batch, beam=beam)
        loss_sum, tokens = loss_sum + float(out["loss"]) * out["tokens"], tokens + out["tokens"]
        language = model.language
        for target, ids in zip(batch["word_targets"], predicted):
            if target is None:
                unpaired += 1
                continue
            words, wrong = words + 1, wrong + (ids != target)
            char_errors, chars = char_errors + edit_distance(target, ids), chars + len(target)
        per_row = [[] for _ in batch["rows"]]
        for row, ids in zip(batch["word_rows"], predicted):
            per_row[row].append(language.decode(ids))
        for row, phonemes in zip(batch["rows"], per_row):
            references.append(" ".join(row["phonemes"].split()))
            hypotheses.append(" ".join(assemble(row["text"], row["spans"], phonemes).split()))
    model.train()
    return {"eval/loss": loss_sum / max(tokens, 1), "eval/word_error": wrong / max(words, 1),
            "eval/cer": char_errors / max(chars, 1), "eval/wer": jiwer.wer(references, hypotheses),
            "eval/unpaired_words": unpaired}, list(zip(references, hypotheses))


def replace_directory(staging, target):
    old = target.with_name(target.name + ".old")
    if target.exists():
        target.rename(old)
    staging.rename(target)
    if old.exists():
        shutil.rmtree(old)


def main():
    args = arguments()
    accelerator = Accelerator(mixed_precision=args.mixed_precision, log_with="tensorboard",
                              project_dir=args.output)
    set_seed(args.seed)
    output = Path(args.output)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        pyinject.listen()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    source = args.resume or (locate(args.init) if args.init else None)
    if args.resume:
        model = G2P.from_checkpoint(args.resume)
        state = json.loads(Path(args.resume, "trainer.json").read_text())
    elif args.init:
        model = G2P.from_checkpoint(source)
        if model.language != args.config.language:
            raise SystemExit(f"--init {args.init} was trained on another inventory than {args.language}")
        state = dict(step=0, epoch=0, batch=0, best=None)
    else:
        model = G2P(args.config.language, backbone=args.config.backbone, width=args.width, layers=args.layers,
                    heads=args.heads, decoder_layers=args.decoder_layers, dropout=args.dropout,
                    max_chars=args.max_chars, max_tokens=args.max_tokens)
        state = dict(step=0, epoch=0, batch=0, best=None)
    settings = model.settings
    tokenizer = None if settings["backbone"] in (None, "none") else AutoTokenizer.from_pretrained(source or settings["backbone"], trust_remote_code=True)
    collate = Collate(model.language, tokenizer, max_chars=settings["max_chars"], max_tokens=settings["max_tokens"],
                      max_word_chars=settings["max_word_chars"], max_phonemes=settings["max_phonemes"])

    train = TSV(args.config.train)
    evaluation = TSV([args.config.eval])
    sampler = Batches(train.lengths(), args.batch_bytes, args.seed, accelerator.process_index, accelerator.num_processes)
    sampler.epoch, sampler.skip = state["epoch"], state["batch"]
    loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers,
                        persistent_workers=args.workers > 0, pin_memory=True)
    eval_loader = DataLoader(evaluation, batch_sampler=Batches(evaluation.lengths(), args.batch_bytes, 0, shuffle=False),
                             collate_fn=collate)

    encoder = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([dict(params=encoder, lr=args.lr), dict(params=heads, lr=args.head_lr)],
                                  weight_decay=args.weight_decay)
    total = args.epochs * len(sampler.epoch_batches())
    schedule = lambda step: min(1.0, (step + 1) / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(step / total, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if args.resume:
        accelerator.load_state(args.resume)
    raw = accelerator.unwrap_model(model)
    accelerator.init_trackers("tensorboard", config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))})
    accelerator.print(f"{sum(p.numel() for p in raw.parameters()) / 1e6:.1f}M parameters; {len(train):,} train rows; "
                      f"{len(sampler.epoch_batches()):,} batches per epoch per rank; {total:,} steps")

    stop = {"requested": False}

    def request_stop(*_):
        stop["requested"] = True
        accelerator.print("Stop requested: checkpointing after this step")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def checkpoint(metrics=None):
        accelerator.wait_for_everyone()
        staging = output / ".last-writing"
        accelerator.save_state(str(staging))
        if accelerator.is_main_process:
            raw.save_config(staging)
            if tokenizer is not None:
                tokenizer.save_pretrained(staging)
            (staging / "trainer.json").write_text(json.dumps(state, indent=2) + "\n")
            replace_directory(staging, output / "last")
            if metrics is not None and state["best"] == metrics["eval/loss"]:
                shutil.rmtree(output / ".best-writing", ignore_errors=True)
                shutil.copytree(output / "last", output / ".best-writing")
                replace_directory(output / ".best-writing", output / "best")
        accelerator.wait_for_everyone()

    def run_eval():
        metrics = None
        if accelerator.is_main_process:
            metrics, samples = evaluate(raw, eval_loader, accelerator.device, args.beam)
            if state["best"] is None or metrics["eval/loss"] < state["best"]:
                state["best"] = metrics["eval/loss"]
            accelerator.log(metrics, step=state["step"])
            (output / "eval-samples.txt").write_text("".join(f"{r}\n{h}\n\n" for r, h in samples))
            tqdm.write(f"step {state['step']}: " + ", ".join(f"{k.split('/')[1]} {v:.4f}" for k, v in metrics.items()))
        checkpoint(metrics)

    model.train()
    deadline = time.monotonic() + args.eval_minutes * 60
    while state["epoch"] < args.epochs:
        sampler.epoch = state["epoch"]
        progress = tqdm(loader, desc=f"epoch {state['epoch']}", initial=state["batch"], total=len(sampler),
                        disable=not accelerator.is_main_process, dynamic_ncols=True)
        running = None
        for batch in progress:
            pyinject.poll()
            out = model(to_device(batch, accelerator.device))
            accelerator.backward(out["loss"])
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"], state["batch"] = state["step"] + 1, state["batch"] + 1
            loss = float(out["loss"])
            running = loss if running is None else 0.98 * running + 0.02 * loss
            progress.set_postfix(loss=f"{running:.4f}", lr=f"{scheduler.get_last_lr()[1]:.2e}")
            if state["step"] % args.log_steps == 0:
                accelerator.log({"train/loss": running, "train/lr": scheduler.get_last_lr()[1]}, step=state["step"])
            if stop["requested"]:
                checkpoint()
                accelerator.end_training()
                return
            if time.monotonic() >= deadline:
                run_eval()
                deadline = time.monotonic() + args.eval_minutes * 60
        state["epoch"], state["batch"] = state["epoch"] + 1, 0
    run_eval()
    accelerator.end_training()


if __name__ == "__main__":
    main()
