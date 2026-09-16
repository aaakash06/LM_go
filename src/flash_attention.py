from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FlashMultiHeadSelfAttention(nn.Module):
    """
    Optimized multi-head self-attention using PyTorch's
    scaled_dot_product_attention.

    PyTorch chooses an appropriate implementation for the current device.

    Despite the name, this should be thought of as an optimized attention
    implementation rather than assuming that a particular CUDA FlashAttention
    kernel is available on every device.
    """

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
                f"d_model ({d_model}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout

        # Fused QKV projection.
        self.qkv_proj = nn.Linear(
            d_model,
            3 * d_model,
            bias=bias,
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)

        # [B, T, 3D]
        # -> [B, T, 3, H, D/H]
        qkv = qkv.view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )

        # -> [3, B, H, T, D/H]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(dim=0)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        # [B, H, T, D/H] -> [B, T, D]
        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, seq_len, self.d_model)

        return self.out_proj(y)