import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceCausalSelfAttention(nn.Module):
    """
    Reference implementation of causal multi-head self-attention.

    This intentionally materializes the full T x T attention matrix.
    It is useful as a correctness/reference implementation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Fused QKV projection.
        # Every attention implementation uses the same parameter layout.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_qkv(self, x):
        B, T, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        return q, k, v

    def forward(self, x):
        B, T, _ = x.shape

        q, k, v = self._split_qkv(x)

        scale = 1.0 / math.sqrt(self.head_dim)

        scores = (q @ k.transpose(-2, -1)) * scale

        # Causal mask.
        mask = torch.triu(
            torch.ones(
                T,
                T,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(mask, float("-inf"))

        attention = F.softmax(scores, dim=-1)

        if self.training and self.dropout > 0:
            attention = F.dropout(
                attention,
                p=self.dropout,
            )

        y = attention @ v

        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, self.d_model)

        return self.out_proj(y)