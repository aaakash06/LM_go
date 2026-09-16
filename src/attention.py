from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            mask,
            float("-inf"),
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        weights = self.dropout(weights)

        y = weights @ v

        y = y.transpose(
            1,
            2,
        ).contiguous()

        y = y.view(
            batch_size,
            seq_len,
            self.d_model,
        )

        return self.out_proj(y)