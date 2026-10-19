"""Transformer layers for Timestral inference in MLX.

Parameter names match the released weights (see `loading.remap_weights` for the one
rename). All arrays use the packed layout: one row per variate of every series in the
batch, `(R, N, D)` with `N` patches. Time attention runs per row over patches; variate
attention runs per patch over the rows of the same series. Rows of a series are
contiguous and every series has the same number of rows `V`, so variate attention runs
on `(B * N, V, D)` groups instead of one `(R, R)` attention over the whole batch.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from timestral_mlx.config import LAYER_KIND_AXES


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.scale = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.scale, self.eps)


class ResidualBlock(nn.Module):
    """`output(silu(hidden(x))) + residual(x)`, used by the patch embedding and the head."""

    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, use_bias: bool) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(input_dims, hidden_dims, bias=use_bias)
        self.output_layer = nn.Linear(hidden_dims, output_dims, bias=use_bias)
        self.residual_layer = nn.Linear(input_dims, output_dims, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.output_layer(nn.silu(self.hidden_layer(x))) + self.residual_layer(x)


class FusedSwiGLU(nn.Module):
    """SwiGLU MLP with one fused input projection; computes `gate * silu(up)`."""

    def __init__(self, embed_dim: int, hidden_dim: int, use_bias: bool) -> None:
        super().__init__()
        self.w12 = nn.Linear(embed_dim, 2 * hidden_dim, bias=use_bias)
        self.w3 = nn.Linear(hidden_dim, embed_dim, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.w12(x), 2, axis=-1)
        return self.w3(gate * nn.silu(up))


class SinCosRoPE(nn.Module):
    """Rotary position embedding (sin/cos concatenation), with partial rotation.

    The inverse-frequency table has `head_dim / 2` entries, zero-padded past
    `rotary_dim / 2`, and `rotate_half` splits the full head. So dimension `i` pairs with
    `i + head_dim / 2` and the padded pairs get the identity rotation. This is not the
    layout of `mx.fast.rope(dims=rotary_dim)`, which pairs `i` with `i + rotary_dim / 2`.
    """

    def __init__(self, dim: int, base: float, rotary_dim: int) -> None:
        super().__init__()
        assert 0 < rotary_dim <= dim and rotary_dim % 2 == 0, f"invalid rotary_dim={rotary_dim} for dim={dim}"
        idx = mx.arange(0, rotary_dim, 2, dtype=mx.float32)
        active = 1.0 / (base ** (idx / rotary_dim))
        # Leading underscore: a constant, not a parameter (it is not in the weights file).
        self._inv_freq = mx.concatenate([active, mx.zeros((dim // 2 - rotary_dim // 2,), dtype=mx.float32)])

    def __call__(self, position_ids: mx.array, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        """Return `(cos, sin)` tables, each `(R, N, head_dim)`, for integer `position_ids` `(R, N)`."""
        freqs = position_ids.astype(mx.float32)[..., None] * self._inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rotary(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, mx.array]:
    """Rotate `q`, `k` shaped `(batch, heads, seq, head_dim)`; the `(batch, seq, head_dim)` tables broadcast over heads."""
    cos = cos[:, None]
    sin = sin[:, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


@dataclass(frozen=True)
class TwoStreamMask:
    """Time mask over the `[Enc ; Dec]` sequence plus its split point.

    Attention with this mask runs two query passes: context queries `[:enc_len]` over
    context keys only, and horizon queries `[enc_len:]` over all keys. This skips the
    always-empty context-to-horizon quadrant.
    """

    mask: mx.array  # (R, 1, S, S) bool, True = attend
    enc_len: int


class Attention(nn.Module):
    """Multi-head attention over the time axis (`is_time=True`) or the variate axis.

    Time attention applies RoPE; variate attention has no positional encoding. With the
    `sdpa_sigmoid` gate, the attention output is multiplied by `sigmoid(x @ W_g)` before
    the output projection.
    """

    def __init__(self, embed_dim: int, num_heads: int, is_time: bool, use_bias: bool, gate: str) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.is_time = is_time
        self.wQKV = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.wO = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        # The weights call this `_gate`; MLX treats underscore names as non-parameters.
        self.gate = nn.Linear(embed_dim, embed_dim, bias=use_bias) if gate == "sdpa_sigmoid" else None

    def _qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Split the fused projection into `(batch, heads, seq, head_dim)` each.

        The weight layout is `(qkv, head_dim, n_heads)`, heads innermost.
        """
        batch, seq, _ = x.shape
        qkv = self.wQKV(x).reshape(batch, seq, 3, self.head_dim, self.num_heads)
        qkv = qkv.transpose(2, 0, 4, 1, 3)
        return qkv[0], qkv[1], qkv[2]

    def _sdpa(self, q: mx.array, k: mx.array, v: mx.array, mask: mx.array | str | None) -> mx.array:
        """`(batch, heads, seq, head_dim)` inputs to `(batch, seq, heads * head_dim)`."""
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        batch, _, seq, _ = out.shape
        return out.transpose(0, 2, 1, 3).reshape(batch, seq, self.embed_dim)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | TwoStreamMask | None,
        cos: mx.array,
        sin: mx.array,
        group_size: int,
    ) -> mx.array:
        rows, num_patches, _ = x.shape
        if self.is_time:
            inputs = x  # (R, N, D)
        else:
            # (R, N, D) -> (B * N, V, D): attention over the V rows of one series at one patch.
            series = rows // group_size
            inputs = x.reshape(series, group_size, num_patches, -1).transpose(0, 2, 1, 3)
            inputs = inputs.reshape(series * num_patches, group_size, -1)
        q, k, v = self._qkv(inputs)
        if self.is_time:
            q, k = _apply_rotary(q, k, cos, sin)
        if isinstance(mask, TwoStreamMask):
            assert self.is_time, "the context/horizon split applies to time attention only"
            n = mask.enc_len
            out_enc = self._sdpa(q[:, :, :n], k[:, :, :n], v[:, :, :n], mask.mask[:, :, :n, :n])
            out_dec = self._sdpa(q[:, :, n:], k, v, mask.mask[:, :, n:, :])
            out = mx.concatenate([out_enc, out_dec], axis=1)
        elif not self.is_time and group_size == 1:
            # One key per query: the softmax weight is exactly 1, so attention returns `v`.
            out = v.transpose(0, 2, 1, 3).reshape(v.shape[0], v.shape[2], self.embed_dim)
        else:
            out = self._sdpa(q, k, v, "causal" if self.is_time and mask is None else mask)
        if self.gate is not None:
            out = out * mx.sigmoid(self.gate(inputs))
        if not self.is_time:
            series = rows // group_size
            out = out.reshape(series, num_patches, group_size, -1).transpose(0, 2, 1, 3).reshape(rows, num_patches, -1)
        return self.wO(out)


class TransformerBlock(nn.Module):
    """The attention(s) of one layer kind, in order, each a pre-norm residual, then one MLP."""

    def __init__(
        self,
        kind: str,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
    ) -> None:
        super().__init__()
        axes = LAYER_KIND_AXES[kind]
        self.attn_norms = [RMSNorm(embed_dim, norm_eps) for _ in axes]
        self.attns = [
            Attention(embed_dim, num_heads, is_time, attn_bias, time_attn_gate if is_time else variate_attn_gate)
            for is_time in axes
        ]
        self.mlp_norm = RMSNorm(embed_dim, norm_eps)
        self.mlp = FusedSwiGLU(embed_dim, mlp_hidden_dim, mlp_bias)

    def __call__(
        self,
        x: mx.array,
        time_mask: mx.array | TwoStreamMask | None,
        variate_mask: mx.array | None,
        cos: mx.array,
        sin: mx.array,
        group_size: int,
    ) -> mx.array:
        for norm, attn in zip(self.attn_norms, self.attns):
            x = x + attn(norm(x), time_mask if attn.is_time else variate_mask, cos, sin, group_size)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(
        self,
        block_schedule: tuple[str, ...],
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        rope_theta: float,
        rope_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
    ) -> None:
        super().__init__()
        self.layers = [
            TransformerBlock(
                kind=kind,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                norm_eps=norm_eps,
                attn_bias=attn_bias,
                mlp_bias=mlp_bias,
                time_attn_gate=time_attn_gate,
                variate_attn_gate=variate_attn_gate,
            )
            for kind in block_schedule
        ]
        self.rope = SinCosRoPE(dim=embed_dim // num_heads, base=rope_theta, rotary_dim=rope_dim)
        self.out_norm = RMSNorm(embed_dim, norm_eps)

    def __call__(
        self,
        x: mx.array,
        time_mask: mx.array | TwoStreamMask | None,
        variate_mask: mx.array | None,
        positions: mx.array,
        group_size: int,
    ) -> mx.array:
        """Run the stack on `x` `(R, N, D)` with integer time `positions` `(R, N)`.

        `variate_mask` is `(B * N, 1, V, V)` for `V = group_size` rows per series, or None.
        """
        assert positions.shape == x.shape[:2], f"positions {tuple(positions.shape)} do not match x {tuple(x.shape)}"
        assert x.shape[0] % group_size == 0, f"{x.shape[0]} rows do not split into groups of {group_size}"
        cos, sin = self.rope(positions, x.dtype)
        for layer in self.layers:
            x = layer(x, time_mask, variate_mask, cos, sin, group_size)
        return self.out_norm(x)
