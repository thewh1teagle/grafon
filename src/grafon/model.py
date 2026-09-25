"""Any encoder's subwords → contextual character states → a small autoregressive decoder per word."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

from .data import BOS, EOS, IGNORE, PAD, Language


def repair_rotary_buffers(encoder):
    """transformers 5 builds remote models on the meta device and leaves non-persistent
    buffers computed in __init__ (NeoBERT's rotary cos/sin tables) as zeros; recompute them."""
    if hasattr(encoder, "freqs_cos") and float(encoder.freqs_cos.abs().max()) == 0.0:
        cos, sin = sys.modules[type(encoder).__module__].precompute_freqs(encoder.config.dim_head, encoder.config.max_length)
        encoder.freqs_cos.copy_(cos.to(encoder.freqs_cos.device, encoder.freqs_cos.dtype))
        encoder.freqs_sin.copy_(sin.to(encoder.freqs_sin.device, encoder.freqs_sin.dtype))


def load_encoder(backbone, encoder_config=None):
    config = AutoConfig.from_pretrained(backbone, trust_remote_code=True)
    kwargs = dict(attn_implementation="sdpa", trust_remote_code=True)
    # Remote-code models (NeoBERT) take no pooler flag.
    if not getattr(config, "auto_map", None):
        kwargs["add_pooling_layer"] = False
    if encoder_config is None:
        encoder = AutoModel.from_pretrained(backbone, **kwargs)
    else:
        config.update(encoder_config)
        encoder = AutoModel.from_config(config, **kwargs)
    repair_rotary_buffers(encoder)
    return encoder


class G2P(nn.Module):
    # Sizes are placeholders until scripts/sweep_capacity.py measures them.
    def __init__(self, language, backbone="dicta-il/neodictabert", width=768, layers=2, heads=12,
                 decoder_layers=2, dropout=0.1, max_chars=1024, max_tokens=512, max_word_chars=64,
                 max_phonemes=64, encoder_config=None):
        super().__init__()
        self.language = language if isinstance(language, Language) else Language.from_dict(language)
        self.encoder = None if backbone in (None, "none") else load_encoder(backbone, encoder_config)
        self.settings = dict(language=self.language.to_dict(), backbone=backbone, width=width, layers=layers, heads=heads,
                             decoder_layers=decoder_layers, dropout=dropout, max_chars=max_chars, max_tokens=max_tokens,
                             max_word_chars=max_word_chars, max_phonemes=max_phonemes,
                             encoder_config=self.encoder.config.to_dict() if self.encoder is not None else None)
        self.max_phonemes = max_phonemes
        if self.encoder is not None:
            # Project at subword length before gathering to the longer character axis.
            self.context_proj = nn.Linear(self.encoder.config.hidden_size, width)
            self.within_token = nn.Embedding(max_chars, width)
        self.characters = nn.Embedding(self.language.char_vocab, width, padding_idx=PAD)
        self.positions = nn.Embedding(max_chars, width)
        self.input_norm = nn.LayerNorm(width)
        block = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, "gelu", batch_first=True, norm_first=True)
        self.stack = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.within_word = nn.Embedding(max_word_chars, width)
        self.phonemes = nn.Embedding(self.language.phoneme_vocab, width, padding_idx=PAD)
        self.steps = nn.Embedding(max_phonemes, width)
        layer = nn.TransformerDecoderLayer(width, heads, width * 4, dropout, "gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, decoder_layers, norm=nn.LayerNorm(width))
        self.output = nn.Linear(width, self.language.phoneme_vocab)

    def states(self, batch):
        char_ids = batch["char_ids"]
        x = self.characters(char_ids) + self.positions(torch.arange(char_ids.size(1), device=char_ids.device))
        if self.encoder is not None:
            hidden = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            context = self.context_proj(hidden.float())
            index = batch["char_to_token"]
            context = context.gather(1, index.clamp_min(0).unsqueeze(-1).expand(-1, -1, context.size(-1)))
            # Characters outside the encoder window keep their letters but get no context.
            x = x + context * (index >= 0).unsqueeze(-1) + self.within_token(batch["within"])
        padding = char_ids == PAD
        padding[:, 0] = False  # an empty row must not be fully masked
        return self.stack(self.input_norm(x), src_key_padding_mask=padding)

    def memory(self, states, word_chars):
        """Each word's character states, [words, chars, width], and their padding mask."""
        flat = states.reshape(-1, states.size(-1))
        memory = flat[word_chars.clamp_min(0)] + self.within_word(torch.arange(word_chars.size(1), device=word_chars.device))
        return memory, word_chars < 0

    def logits(self, memory, padding, tokens):
        y = self.phonemes(tokens) + self.steps(torch.arange(tokens.size(1), device=tokens.device))
        causal = nn.Transformer.generate_square_subsequent_mask(tokens.size(1), device=tokens.device, dtype=y.dtype)
        y = self.decoder(y, memory, tgt_mask=causal, tgt_is_causal=True, memory_key_padding_mask=padding)
        return self.output(y)

    def forward(self, batch):
        states = self.states(batch)
        supervised = batch["supervised"]
        if supervised.numel() == 0:
            # Keep every parameter in the graph so DDP's gradient sync still matches.
            return dict(loss=sum(p.sum() for p in self.parameters()) * 0.0, tokens=0)
        memory, padding = self.memory(states, batch["word_chars"][supervised])
        logits = self.logits(memory, padding, batch["decoder_input"])
        labels = batch["labels"]
        loss = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=IGNORE)
        return dict(loss=loss, tokens=int((labels != IGNORE).sum()))

    def split_heads(self, x):
        heads = self.decoder.layers[0].self_attn.num_heads
        return x.view(x.size(0), x.size(1), heads, -1).transpose(1, 2)

    def cross_cache(self, memory):
        """Per-layer cross-attention keys and values; the word's characters never change."""
        width, cache = memory.size(-1), []
        for layer in self.decoder.layers:
            attn = layer.multihead_attn
            k = F.linear(memory, attn.in_proj_weight[width:2 * width], attn.in_proj_bias[width:2 * width])
            v = F.linear(memory, attn.in_proj_weight[2 * width:], attn.in_proj_bias[2 * width:])
            cache.append((self.split_heads(k), self.split_heads(v)))
        return cache

    def step(self, token, position, cross, past, attend):
        """One decoding step with a KV cache; the same weights and math as `logits` in eval mode."""
        y = self.phonemes(token) + self.steps.weight[position]
        width = y.size(-1)
        for i, layer in enumerate(self.decoder.layers):
            attn = layer.self_attn
            q, k, v = (self.split_heads(t) for t in F.linear(layer.norm1(y), attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1))
            if past[i] is not None:
                k, v = torch.cat([past[i][0], k], 2), torch.cat([past[i][1], v], 2)
            past[i] = (k, v)
            y = y + attn.out_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(y.shape))
            attn = layer.multihead_attn
            q = self.split_heads(F.linear(layer.norm2(y), attn.in_proj_weight[:width], attn.in_proj_bias[:width]))
            context = F.scaled_dot_product_attention(q, *cross[i], attn_mask=attend)
            y = y + attn.out_proj(context.transpose(1, 2).reshape(y.shape))
            y = y + layer.linear2(layer.activation(layer.linear1(layer.norm3(y))))
        return self.output(self.decoder.norm(y))[:, -1]

    @torch.no_grad()
    def generate(self, batch, beam=1, words=None):
        """Phoneme ids for every word of the batch (or the given word indices)."""
        states = self.states(batch)
        word_chars = batch["word_chars"] if words is None else batch["word_chars"][words]
        if word_chars.size(0) == 0:
            return []
        memory, padding = self.memory(states, word_chars)
        count = memory.size(0)
        memory, padding = memory.repeat_interleave(beam, 0), padding.repeat_interleave(beam, 0)
        cross, attend = self.cross_cache(memory), ~padding[:, None, None, :]
        past = [None] * len(self.decoder.layers)
        tokens = torch.full((count * beam, 1), BOS, dtype=torch.long, device=memory.device)
        scores = torch.zeros(count, beam, device=memory.device)
        scores[:, 1:] = float("-inf")  # all beams start identical; keep one
        done = torch.zeros(count * beam, dtype=torch.bool, device=memory.device)
        rows = torch.arange(count, device=memory.device).unsqueeze(1)
        for position in range(self.max_phonemes - 1):
            logp = F.log_softmax(self.step(tokens[:, -1:], position, cross, past, attend).float(), -1)
            logp[:, PAD] = float("-inf")
            logp[:, BOS] = float("-inf")
            # A finished beam only extends with padding, at no cost.
            logp[done] = float("-inf")
            logp[done, PAD] = 0.0
            vocab = logp.size(-1)
            top, choice = (scores.view(-1, 1) + logp).view(count, beam * vocab).topk(beam, -1)
            origin, token = choice // vocab, choice % vocab
            source = (rows * beam + origin).view(-1)
            tokens = torch.cat([tokens[source], token.view(-1, 1)], 1)
            if beam > 1:
                past = [(k[source], v[source]) for k, v in past]
            done = done[source] | (token.view(-1) == EOS)
            scores = top
            if done.all():
                break
        best = tokens.view(count, beam, -1)[:, 0, 1:].tolist()
        return [ids[:ids.index(EOS)] if EOS in ids else ids for ids in best]

    def save_config(self, directory):
        Path(directory, "config.json").write_text(json.dumps(self.settings, ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def from_checkpoint(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        model = cls(**json.loads((directory / "config.json").read_text()))
        model.load_state_dict(load_file(str(directory / "model.safetensors")))
        return model
