from __future__ import annotations

import math

import torch
from torch import Tensor, nn



class MultiHeadSelfAttention(nn.Module):
    """
    Standard scaled dot-product multi-head self-attention.

    Input:
        x: [batch, sequence_length, d_model]

    Output:
        [batch, sequence_length, d_model]

    This implementation explicitly constructs the attention matrix and is
    intended primarily as a reference implementation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # [B, T, D] -> [B, H, T, D/H]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # [B, H, T, D/H] @ [B, H, D/H, T]
        # -> [B, H, T, T]
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(
                    seq_len,
                    seq_len,
                    device=x.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )

            scores = scores.masked_fill(causal_mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # [B, H, T, T] @ [B, H, T, D/H]
        # -> [B, H, T, D/H]
        y = attention_weights @ v

        # [B, H, T, D/H] -> [B, T, D]
        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, seq_len, self.d_model)

        return self.out_proj(y)