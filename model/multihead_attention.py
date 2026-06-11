"""
Prototype implementation of multi-head attention in PyTorch.

The module is intentionally standalone so it can be tested without changing the
current DeepAR training path. It supports self-attention and cross-attention on
batch-first tensors shaped as [batch_size, seq_len, embed_dim].
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a boolean mask that hides future positions."""
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention.

    Args:
        embed_dim: Input and output feature dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability applied to attention weights.
        bias: Whether to add bias in linear projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run attention.

        Args:
            query: Tensor with shape [batch_size, query_len, embed_dim].
            key: Optional tensor with shape [batch_size, key_len, embed_dim].
                Defaults to query for self-attention.
            value: Optional tensor with shape [batch_size, key_len, embed_dim].
                Defaults to key.
            attn_mask: Optional bool mask. True positions are ignored. Supported
                shapes are [query_len, key_len], [batch_size, query_len, key_len],
                or [batch_size, num_heads, query_len, key_len].
            need_weights: If True, also return attention weights averaged across
                heads with shape [batch_size, query_len, key_len].

        Returns:
            output: Tensor with shape [batch_size, query_len, embed_dim].
            weights: Attention weights if requested, otherwise None.
        """
        key = query if key is None else key
        value = key if value is None else value
        self._validate_inputs(query, key, value)

        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        q = self._shape_projection(self.q_proj(query), batch_size, query_len)
        k = self._shape_projection(self.k_proj(key), batch_size, key_len)
        v = self._shape_projection(self.v_proj(value), batch_size, key_len)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            mask = self._expand_mask(attn_mask, batch_size, query_len, key_len)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, query_len, self.embed_dim)
        output = self.out_proj(context)

        if need_weights:
            return output, weights.mean(dim=1)
        return output, None

    def _shape_projection(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            if tensor.dim() != 3:
                raise ValueError(f"{name} must be a 3D tensor")
            if tensor.shape[-1] != self.embed_dim:
                raise ValueError(
                    f"{name} feature dimension must equal embed_dim={self.embed_dim}"
                )

        if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
            raise ValueError("query, key, and value must share batch_size")
        if key.shape[1] != value.shape[1]:
            raise ValueError("key and value must share sequence length")

    def _expand_mask(
        self,
        attn_mask: torch.Tensor,
        batch_size: int,
        query_len: int,
        key_len: int,
    ) -> torch.Tensor:
        mask = attn_mask.to(dtype=torch.bool)

        if mask.dim() == 2:
            expected = (query_len, key_len)
            if mask.shape != expected:
                raise ValueError(f"2D attn_mask must have shape {expected}")
            return mask.unsqueeze(0).unsqueeze(0)

        if mask.dim() == 3:
            expected = (batch_size, query_len, key_len)
            if mask.shape != expected:
                raise ValueError(f"3D attn_mask must have shape {expected}")
            return mask.unsqueeze(1)

        if mask.dim() == 4:
            expected = (batch_size, self.num_heads, query_len, key_len)
            if mask.shape != expected:
                raise ValueError(f"4D attn_mask must have shape {expected}")
            return mask

        raise ValueError("attn_mask must be 2D, 3D, or 4D")


def demo() -> None:
    """Run a small shape check for the prototype."""
    torch.manual_seed(230)
    batch_size = 2
    seq_len = 5
    embed_dim = 16
    num_heads = 4

    x = torch.randn(batch_size, seq_len, embed_dim)
    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    mask = causal_mask(seq_len, device=x.device)
    output, weights = attention(x, attn_mask=mask, need_weights=True)

    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(output.shape))
    print("attention weights shape:", tuple(weights.shape))


if __name__ == "__main__":
    demo()
