from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions].to(query.device)
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        # rope_scaling: dict | None = None,
):
    # assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


def get_rope_llama(
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
):
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)


def apply_multimodal_rotary_emb(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embedding for multimodal inputs (MRoPE).
    cos and sin can have either shape [3, seq_len, head_dim] or [seq_len, head_dim].
    x has shape [seq_len, num_heads, head_dim].
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)

    if cos.ndim == 3:
        cos_expanded = cos.transpose(0, 1).unsqueeze(1)
        sin_expanded = sin.transpose(0, 1).unsqueeze(1)

        cos_t = cos_expanded[:, :, 0, :]
        cos_h = cos_expanded[:, :, 1, :]
        cos_w = cos_expanded[:, :, 2, :]

        sin_t = sin_expanded[:, :, 0, :]
        sin_h = sin_expanded[:, :, 1, :]
        sin_w = sin_expanded[:, :, 2, :]

        y1 = x1 * (cos_t + cos_h + cos_w) / 3 - x2 * (sin_t + sin_h + sin_w) / 3
        y2 = x2 * (cos_t + cos_h + cos_w) / 3 + x1 * (sin_t + sin_h + sin_w) / 3
    else:
        cos_expanded = cos.unsqueeze(1)
        sin_expanded = sin.unsqueeze(1)

        y1 = x1 * cos_expanded - x2 * sin_expanded
        y2 = x2 * cos_expanded + x1 * sin_expanded

    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class MropeRotaryEmbedding(nn.Module):

    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: float,
            rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        positions has shape [3, seq_len] for (t, h, w) dimensions.
        Returns cos, sin each with shape [3, seq_len, rotary_dim].
        """
        device = positions.device
        dtype = positions.dtype

        cos_list = []
        sin_list = []

        for dim_idx in range(3):
            pos = positions[dim_idx]
            t = pos.float()
            freqs = torch.einsum("i,j -> ij", t, self.inv_freq.to(dtype))
            cos_dim = freqs.cos()
            sin_dim = freqs.sin()
            cos_list.append(cos_dim)
            sin_list.append(sin_dim)

        cos = torch.stack(cos_list, dim=0)
        sin = torch.stack(sin_list, dim=0)

        return cos, sin

    def forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MRoPE.
        positions: [3, seq_len] or [seq_len] for single-dimension positions
        query: [seq_len, num_heads, head_dim]
        key: [seq_len, num_kv_heads, head_dim]
        """
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)

        cos, sin = self._get_cos_sin(positions)
        query = apply_multimodal_rotary_emb(query, cos, sin)
        key = apply_multimodal_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rotary_emb_mrope(
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        rope_scaling: dict | None = None,
):
    rotary_emb = MropeRotaryEmbedding(head_size, rotary_dim, max_position, base, rope_scaling)
    return rotary_emb
