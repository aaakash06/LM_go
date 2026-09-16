from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .flash_attention import (
    FlashMultiHeadSelfAttention,
)


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4

    d_ff: int = 1024

    dropout: float = 0.0


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(d_model)
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        variance = x.float().pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x = x * torch.rsqrt(
            variance + self.eps
        )

        return self.weight * x


class FeedForward(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.w1 = nn.Linear(
            d_model,
            d_ff,
        )

        self.w2 = nn.Linear(
            d_model,
            d_ff,
        )

        self.w3 = nn.Linear(
            d_ff,
            d_model,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x = F.silu(
            self.w1(x)
        ) * self.w2(x)

        return self.dropout(
            self.w3(x)
        )


class TransformerBlock(nn.Module):

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        self.norm1 = RMSNorm(
            config.d_model
        )

        self.attention = (
            FlashMultiHeadSelfAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
        )

        self.norm2 = RMSNorm(
            config.d_model
        )

        self.feed_forward = FeedForward(
            config.d_model,
            config.d_ff,
            config.dropout,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


class TransformerLM(nn.Module):

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

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            config.d_model
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # Weight tying.
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(self._init_weights)

    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:

        if isinstance(
            module,
            nn.Linear,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):
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
                "Sequence is longer than max_seq_len"
            )

        positions = torch.arange(
            seq_len,
            device=tokens.device,
        )

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(
                positions
            )
        )

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:

        self.eval()

        for _ in range(max_new_tokens):

            tokens_cond = tokens[
                :,
                -self.config.max_seq_len :,
            ]

            logits, _ = self(
                tokens_cond
            )

            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:
                values, _ = torch.topk(
                    logits,
                    min(
                        top_k,
                        logits.size(-1),
                    ),
                )

                threshold = values[:, [-1]]

                logits = torch.where(
                    logits < threshold,
                    torch.full_like(
                        logits,
                        float("-inf"),
                    ),
                    logits,
                )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            tokens = torch.cat(
                [
                    tokens,
                    next_token,
                ],
                dim=1,
            )

        return tokens