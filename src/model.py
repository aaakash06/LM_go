from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .flash_attention import FlashMultiHeadSelfAttention


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int

    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8

    d_ff: int | None = None
    dropout: float = 0.0

    bias: bool = True

    # RMSNorm is common in modern LMs.
    # Set to False if you want LayerNorm.
    use_rmsnorm: bool = True


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # Accumulate norm in float32 for numerical stability.
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.eps)

        return self.weight * x


class FeedForward(nn.Module):
    """
    SwiGLU feed-forward network.

    Output dimensionality is d_model.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.w1 = nn.Linear(
            d_model,
            d_ff,
            bias=bias,
        )

        self.w2 = nn.Linear(
            d_model,
            d_ff,
            bias=bias,
        )

        self.w3 = nn.Linear(
            d_ff,
            d_model,
            bias=bias,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU:
        #
        # SiLU(xW1) * (xW2)
        #
        # followed by W3.
        hidden = torch.nn.functional.silu(self.w1(x))
        hidden = hidden * self.w2(x)

        return self.dropout(self.w3(hidden))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        d_ff = config.d_ff

        if d_ff is None:
            # Reasonable default for a SwiGLU model.
            d_ff = int(8 * config.d_model / 3)

            # Round to a convenient multiple.
            d_ff = ((d_ff + 63) // 64) * 64

        if config.use_rmsnorm:
            norm_factory = lambda: RMSNorm(config.d_model)
        else:
            norm_factory = lambda: nn.LayerNorm(
                config.d_model,
                eps=1e-5,
            )

        self.norm1 = norm_factory()

        self.attention = FlashMultiHeadSelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            bias=config.bias,
            causal=True,
        )

        self.norm2 = norm_factory()

        self.feed_forward = FeedForward(
            d_model=config.d_model,
            d_ff=d_ff,
            dropout=config.dropout,
            bias=config.bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm Transformer.
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))

        return x


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer language model.

    Input:
        tokens: [B, T]

    Output:
        logits: [B, T, vocab_size]
    """

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
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

        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        if config.use_rmsnorm:
            self.final_norm = RMSNorm(config.d_model)
        else:
            self.final_norm = nn.LayerNorm(
                config.d_model,
                eps=1e-5,
            )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # Weight tying between input embeddings and output projection.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
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

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        batch_size, seq_len = tokens.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_seq_len={self.config.max_seq_len}"
            )

        positions = torch.arange(
            seq_len,
            device=tokens.device,
        )

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
        )

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            input_tokens = tokens[:, -self.config.max_seq_len :]

            logits, _ = self(input_tokens)

            next_token_logits = logits[:, -1, :]

            next_token_logits = next_token_logits / temperature

            if top_k is not None:
                values, _ = torch.topk(
                    next_token_logits,
                    min(top_k, next_token_logits.size(-1)),
                )

                threshold = values[:, [-1]]

                next_token_logits = torch.where(
                    next_token_logits < threshold,
                    torch.full_like(
                        next_token_logits,
                        float("-inf"),
                    ),
                    next_token_logits,
                )

            probabilities = torch.softmax(
                next_token_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1,
            )

        return tokens