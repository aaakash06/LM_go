from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ReferenceCausalSelfAttention
from .flash_attention import SDPACAusalSelfAttention
from .chunked_attention import ChunkedCausalSelfAttention


@dataclass
class TransformerConfig:
    vocab_size: int

    max_seq_len: int = 128

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024

    dropout: float = 0.0

    attention_type: str = "sdpa"
    chunk_size: int = 128


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(
            variance + self.eps
        )

        return self.weight * x


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(
            F.silu(self.w1(x)) * self.w2(x)
        )


def build_attention(config: TransformerConfig):
    if config.attention_type == "reference":
        return ReferenceCausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

    if config.attention_type == "sdpa":
        return SDPACAusalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

    if config.attention_type == "chunked":
        return ChunkedCausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            chunk_size=config.chunk_size,
        )

    raise ValueError(
        f"Unknown attention_type: {config.attention_type}"
    )


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.norm1 = RMSNorm(config.d_model)

        self.attention = build_attention(config)

        self.norm2 = RMSNorm(config.d_model)

        self.mlp = SwiGLU(
            config.d_model,
            config.d_ff,
        )

    def forward(self, x):
        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # Weight tying.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape

        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"max_seq_len={self.config.max_seq_len}"
            )

        positions = torch.arange(
            T,
            device=input_ids.device,
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = input_ids[
                :, -self.config.max_seq_len:
            ]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            if temperature <= 0:
                raise ValueError(
                    "temperature must be positive"
                )

            logits = logits / temperature

            if top_k is not None:
                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1)),
                )

                cutoff = values[:, [-1]]

                logits = torch.where(
                    logits < cutoff,
                    torch.full_like(
                        logits,
                        float("-inf"),
                    ),
                    logits,
                )

            probs = F.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

        return input_ids


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )