import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPACAusalSelfAttention(nn.Module):
    """
    Causal self-attention using PyTorch's scaled_dot_product_attention.

    On supported hardware/backends PyTorch may select a fused implementation.
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
        q, k, v = self._split_qkv(x)

        dropout_p = self.dropout if self.training else 0.0

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous()
        y = y.view(x.shape[0], x.shape[1], self.d_model)

        return self.out_proj(y)