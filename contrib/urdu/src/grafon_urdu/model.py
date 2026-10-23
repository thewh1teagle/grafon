"""Small encoder-decoder transformer for contextual Urdu G2P.

No BERT, XLM-R, LLM, or pretrained multilingual tokenizer weights are used.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn


@dataclass
class ModelConfig:
    source_vocab_size: int
    target_vocab_size: int
    d_model: int = 192
    nhead: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.10


class SinusoidalPositions(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(x.shape[1], dtype=torch.float32, device=x.device)[:, None]
        dim = x.shape[2]
        freqs = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=x.device)
                          * (-math.log(10000.0) / dim))
        pe = torch.zeros((x.shape[1], dim), device=x.device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(steps * freqs)
        pe[:, 1::2] = torch.cos(steps * freqs[:pe[:, 1::2].shape[1]])
        return x + pe.to(dtype=x.dtype).unsqueeze(0)


class CompactG2P(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.src_emb = nn.Embedding(cfg.source_vocab_size, cfg.d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(cfg.target_vocab_size, cfg.d_model, padding_idx=0)
        self.positions = SinusoidalPositions()
        self.net = nn.Transformer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            num_encoder_layers=cfg.encoder_layers, num_decoder_layers=cfg.decoder_layers,
            dim_feedforward=cfg.feedforward_dim, dropout=cfg.dropout,
            batch_first=True, norm_first=False,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, cfg.target_vocab_size, bias=False)
        self.scale = math.sqrt(cfg.d_model)

    def memory(self, src: torch.Tensor) -> torch.Tensor:
        x = self.positions(self.src_emb(src) * self.scale)
        return self.net.encoder(x, src_key_padding_mask=src.eq(0))

    def logits(self, tgt_in: torch.Tensor, memory: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        y = self.positions(self.tgt_emb(tgt_in) * self.scale)
        length = tgt_in.shape[1]
        causal = torch.triu(torch.ones(length, length, device=tgt_in.device, dtype=torch.bool), 1)
        y = self.net.decoder(
            y, memory, tgt_mask=causal,
            tgt_key_padding_mask=tgt_in.eq(0), memory_key_padding_mask=src.eq(0)
        )
        return self.output(self.norm(y))

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.logits(tgt_in, self.memory(src), src)

    @torch.inference_mode()
    def generate(self, src: torch.Tensor, *, bos: int = 2, eos: int = 3,
                 max_new_tokens: int = 480) -> list[list[int]]:
        if src.ndim != 2:
            raise ValueError("src needs shape [batch, seq]")
        self.eval()
        memory = self.memory(src)
        output = torch.full((src.shape[0], 1), bos, device=src.device, dtype=torch.long)
        done = torch.zeros(src.shape[0], device=src.device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            next_logits = self.logits(output, memory, src)[:, -1]
            next_logits[:, :3] = -torch.inf  # pad/unk/bos are not IPA outputs
            next_token = next_logits.argmax(-1)
            next_token = torch.where(done, eos, next_token)
            output = torch.cat((output, next_token[:, None]), dim=1)
            done |= next_token.eq(eos)
            if bool(done.all()):
                break
        return output[:, 1:].cpu().tolist()

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
