import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkedCausalSelfAttention(nn.Module):
    """
    Memory-efficient causal attention implemented by processing
    query positions in chunks.

    The mathematical operation is still:

        softmax(QK^T / sqrt(d)) V

    but instead of materializing all T x T scores simultaneously,
    only a chunk_size x T score matrix is materialized at a time.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        chunk_size: int = 128,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.chunk_size = chunk_size

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

        outputs = []

        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)

            q_chunk = q[:, :, start:end, :]

            # [B, H, chunk, T]
            scores = torch.matmul(
                q_chunk,
                k.transpose(-2, -1),
            ) * scale

            # Query positions are:
            # start, start + 1, ..., end - 1
            query_positions = torch.arange(
                start,
                end,
                device=x.device,
            ).view(1, 1, -1, 1)

            key_positions = torch.arange(
                T,
                device=x.device,
            ).view(1, 1, 1, -1)

            # True where key position is in the future.
            future_mask = key_positions > query_positions

            scores = scores.masked_fill(
                future_mask,
                float("-inf"),
            )

            attention = F.softmax(scores, dim=-1)

            if self.training and self.dropout > 0:
                attention = F.dropout(
                    attention,
                    p=self.dropout,
                )

            y_chunk = torch.matmul(attention, v)

            outputs.append(y_chunk)

        y = torch.cat(outputs, dim=2)

        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, self.d_model)

        return self.out_proj(y)