"""The Timestral network and its single-block forecast primitive, in MLX.

The model normalizes each row causally per patch, embeds patches from `[values,
missing]` channels, runs a time/variate transformer over `[Enc ; Dec]` tokens (context
patches followed by learned forecast-query tokens), and reads quantiles for every
horizon time step from the decoder tokens.

Rows are packed per series: `num_series` series of `V = R / num_series` contiguous rows
each (targets first, then covariates). Row roles are host-side NumPy metadata, so
choosing a code path never waits on the GPU.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from timestral_mlx.config import TimestralConfig
from timestral_mlx.layers import ResidualBlock, Transformer, TwoStreamMask
from timestral_mlx.masks import (
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_positions,
    build_two_stream_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    group_first_observed_patch,
)
from timestral_mlx.scaler import (
    FP32_MAX,
    apply_affine_asinh_masked,
    causal_patch_loc_scale,
    denormalize_quantiles,
    instance_norm_stats,
)

ROLE_TARGET = 0
ROLE_PAST_COV = 1
ROLE_KNOWN_COV = 2


class PatchRepresentation(nn.Module):
    """Patch embedding over per-step `[values, missing]` channels, plus the forecast-query token."""

    def __init__(self, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = ResidualBlock(2 * patch_size, 4 * d_model, d_model, use_bias=True)
        self.forecast_query = mx.zeros((d_model,))

    def __call__(self, scaled: mx.array, observed: mx.array) -> mx.array:
        """`(R, T)` normalized values and observation mask to `(R, T / patch_size, d_model)`."""
        r, t = scaled.shape
        n = t // self.patch_size
        values = scaled.reshape(r, n, self.patch_size)
        missing = (~observed).reshape(r, n, self.patch_size).astype(scaled.dtype)
        x = mx.concatenate([values, missing], axis=-1).astype(self.patch_proj.hidden_layer.weight.dtype)
        return self.patch_proj(x)

    def forecast_tokens(self, rows: int, n_patches: int) -> mx.array:
        return mx.broadcast_to(self.forecast_query.reshape(1, 1, -1), (rows, n_patches, self.forecast_query.shape[0]))


def _softplus(x: mx.array) -> mx.array:
    """PyTorch's softplus (`beta=1`, `threshold=20`): linear above the threshold."""
    return mx.where(x > 20.0, x, mx.log1p(mx.exp(mx.minimum(x, 20.0))))


def monotone_quantiles(raw: mx.array, levels: tuple[float, ...], width_factor: float) -> mx.array:
    """Non-crossing quantiles from raw head outputs along the last axis (`monotone_cumsum`).

    The median channel is free. The others fan out from it as
    `median -/+ cumsum(width_factor * dp * softplus(raw))`, `dp` being the probability gap
    to the next level inward, so unequal knot spacing is respected.
    """
    m = levels.index(0.5)
    widths = [0.0] * len(levels)
    for i in range(m + 1, len(levels)):
        widths[i] = levels[i] - levels[i - 1]
    for i in range(m):
        widths[i] = levels[i + 1] - levels[i]
    inc = width_factor * mx.array(widths, dtype=raw.dtype) * _softplus(raw)
    median = raw[..., m : m + 1]
    up = mx.cumsum(inc[..., m + 1 :], axis=-1)
    down = mx.cumsum(inc[..., :m], axis=-1, reverse=True)
    return mx.clip(mx.concatenate([median - down, median, median + up], axis=-1), -FP32_MAX, FP32_MAX)


class QuantileHead(nn.Module):
    """Maps each decoder token to `patch_size` monotone quantile vectors in normalized space."""

    def __init__(self, config: TimestralConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.num_quantiles = config.num_quantiles
        self.config = config
        self.projection = ResidualBlock(
            config.d_model, 4 * config.d_model, config.patch_size * config.num_quantiles, use_bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        """`(R, N, D)` to normalized quantiles `(R, N * patch_size, Q)` in float32."""
        raw = self.projection(x)
        *leading, n_patches, _ = raw.shape
        preds = raw.reshape(*leading, n_patches, self.patch_size, self.num_quantiles).astype(mx.float32)
        preds = monotone_quantiles(preds, self.config.quantile_knots, self.config.quantile_width_factor)
        return preds.reshape(*leading, n_patches * self.patch_size, self.num_quantiles)


class TimestralModel(nn.Module):
    def __init__(self, config: TimestralConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.representation = PatchRepresentation(c.patch_size, c.d_model)
        self.transformer = Transformer(
            block_schedule=c.block_schedule,
            embed_dim=c.d_model,
            num_heads=c.num_heads,
            mlp_hidden_dim=c.mlp_hidden_dim,
            rope_theta=c.rope_theta,
            rope_dim=c.rope_dim,
            norm_eps=c.norm_eps,
            attn_bias=c.attn_bias,
            mlp_bias=c.mlp_bias,
            time_attn_gate=c.time_attn_gate,
            variate_attn_gate=c.variate_attn_gate,
        )
        self.head = QuantileHead(c)

    def scale(self, inputs: mx.array, observed: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Causal per-patch normalization of packed `(R, T)` rows.

        Returns `(scaled, loc, scale)`, each `(R, T)`.
        """
        c = self.config
        loc, scale = causal_patch_loc_scale(inputs, observed, c.patch_size, c.minimum_scale)
        scaled = apply_affine_asinh_masked(inputs, loc, scale, observed, use_asinh=c.use_asinh)
        return scaled, loc, scale

    def _known_cov_scaling(
        self, inputs: mx.array, observed: mx.array, scaled: mx.array, loc: mx.array, scale: mx.array, known_rows: np.ndarray
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Normalize known-future covariate rows with non-causal whole-row statistics.

        Indicator rows (values in {0, 1}) skip the asinh squash.
        """
        if not known_rows.any():
            return scaled, loc, scale
        loc_c, scale_c, is_binary = instance_norm_stats(inputs, observed, minimum_scale=self.config.minimum_scale)
        known_scaled = apply_affine_asinh_masked(inputs, loc_c, scale_c, observed, use_asinh=False)
        if self.config.use_asinh:
            known_scaled = mx.where(observed & ~is_binary, mx.arcsinh(known_scaled), known_scaled)
        sel = mx.array(known_rows)[:, None]
        return mx.where(sel, known_scaled, scaled), mx.where(sel, loc_c, loc), mx.where(sel, scale_c, scale)

    def _run_two_stream(
        self,
        x_enc: mx.array,
        x_dec: mx.array,
        time_mask: mx.array,
        variate_mask: mx.array | None,
        positions: mx.array,
        group_size: int,
    ) -> mx.array:
        """Transformer over `[Enc ; Dec]`, then the head on the decoder tokens (normalized space)."""
        x = mx.concatenate([x_enc, x_dec], axis=1)
        mask: mx.array | TwoStreamMask = time_mask
        if self.config.split_context_horizon_attn:
            mask = TwoStreamMask(mask=time_mask, enc_len=x_enc.shape[1])
        h = self.transformer(x, mask, variate_mask, positions, group_size)
        return self.head(h[:, x_enc.shape[1] :, :])

    def predict_block(
        self,
        inputs: mx.array,
        observed: mx.array,
        num_series: int,
        horizon_patches: int,
        role_ids: np.ndarray | None = None,
        known_future: mx.array | None = None,
        known_future_mask: mx.array | None = None,
    ) -> mx.array:
        """Forecast one block of `horizon_patches` patches after a packed context.

        Args:
            inputs: `(R, T)` context values, `T` a multiple of the patch size.
            observed: `(R, T)` observation mask.
            num_series: Number of series `B`; series `b` owns the `V = R / B` contiguous
                rows `b * V .. b * V + V - 1`, which attend each other.
            horizon_patches: Number of patches to forecast.
            role_ids: `(R,)` NumPy row roles (`ROLE_TARGET`, `ROLE_PAST_COV`,
                `ROLE_KNOWN_COV`), or None when every row is a target.
            known_future: `(R, horizon_patches * patch_size)` known-future covariate
                values over this block; zeros on other rows.
            known_future_mask: Observation mask for `known_future`.

        Returns:
            Raw-space quantiles `(R, horizon_patches * patch_size, Q)`, non-decreasing along Q.
        """
        assert inputs.ndim == 2, f"expected packed (R, T) inputs, got {tuple(inputs.shape)}"
        r = inputs.shape[0]
        assert num_series >= 1 and r % num_series == 0, f"{r} rows do not split into {num_series} equal series"
        group_size = r // num_series
        if role_ids is None or not self.config.covariate_enabled or bool((role_ids == ROLE_TARGET).all()):
            return self._predict_context_only(inputs, observed, group_size, horizon_patches)
        role_ids = np.asarray(role_ids)
        assert role_ids.shape == (r,), f"role_ids must be ({r},), got {role_ids.shape}"
        horizon_len = horizon_patches * self.config.patch_size
        if known_future is None:
            known_future = mx.zeros((r, horizon_len), dtype=inputs.dtype)
            known_future_mask = mx.zeros((r, horizon_len), dtype=mx.bool_)
        assert known_future_mask is not None, "known_future_mask is required alongside known_future"
        assert known_future.shape == (r, horizon_len), f"known_future must be ({r}, {horizon_len}), got {tuple(known_future.shape)}"
        full_inputs = mx.concatenate([inputs, known_future], axis=-1)
        full_observed = mx.concatenate([observed, known_future_mask], axis=-1)
        return self._predict_window(full_inputs, full_observed, group_size, horizon_patches, role_ids)

    def _predict_context_only(self, inputs: mx.array, observed: mx.array, group_size: int, horizon_patches: int) -> mx.array:
        """No covariates: encoder over the context, `h` forecast-query tokens after it."""
        c = self.config
        ps = c.patch_size
        rows, t = inputs.shape
        assert t % ps == 0, f"context length {t} must be a multiple of patch_size {ps}"
        ll, h = t // ps, horizon_patches
        assert h >= 1, f"horizon_patches must be >= 1, got {h}"

        scaled, loc, scale = self.scale(inputs, observed)
        x_enc = self.representation(scaled, observed)
        x_dec = self.representation.forecast_tokens(rows, h)

        obs_per_patch = observed.reshape(rows, ll, ps).any(axis=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        time_mask = build_forecast_time_mask(ll, h, key_valid)
        variate_mask = build_two_stream_variate_mask(key_valid, h, group_size)
        first_obs = group_first_observed_patch(obs_per_patch, group_size)
        positions = build_forecast_positions(ll, h, first_obs)

        preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions, group_size)
        anchor_loc = loc[:, (ll - 1) * ps : (ll - 1) * ps + 1]
        anchor_scale = scale[:, (ll - 1) * ps : (ll - 1) * ps + 1]
        return denormalize_quantiles(preds, anchor_loc[..., None], anchor_scale[..., None], c.use_asinh)

    def _predict_window(
        self, full_inputs: mx.array, full_observed: mx.array, group_size: int, horizon_patches: int, role_ids: np.ndarray
    ) -> mx.array:
        """With covariates: encoder over context plus horizon, so known-future values are keys.

        Target and past-only rows keep a blank horizon. The time and variate masks keep
        target information out of covariate rows and keep any row's future out of the
        targets.
        """
        c = self.config
        ps = c.patch_size
        rows, w = full_inputs.shape
        assert w % ps == 0, f"window length {w} must be a multiple of patch_size {ps}"
        h = horizon_patches
        assert 1 <= h <= c.block_size, f"decode block {h} must be in [1, {c.block_size}] with covariates"
        n = w // ps
        ll = n - h
        assert ll >= 1, f"context must have at least one patch (window {n}, horizon {h})"
        target_np = role_ids == ROLE_TARGET
        known_np = role_ids == ROLE_KNOWN_COV
        target_rows, known_rows = mx.array(target_np), mx.array(known_np)

        scaled, loc, scale = self.scale(full_inputs, full_observed)
        scaled, loc, scale = self._known_cov_scaling(full_inputs, full_observed, scaled, loc, scale, known_np)
        x_enc = self.representation(scaled, full_observed)
        x_dec = self.representation.forecast_tokens(rows, n)

        obs_per_patch = full_observed.reshape(rows, n, ps).any(axis=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        s_rows = mx.full((rows,), ll, dtype=mx.int32)
        time_mask = build_two_stream_time_mask(s_rows, n, c.block_size, key_valid, bidirectional_rows=known_rows)
        variate_mask = build_two_stream_variate_mask(
            key_valid, n, group_size, target_rows=target_rows, bidirectional_rows=known_rows
        )
        first_obs = group_first_observed_patch(obs_per_patch, group_size)
        positions = build_two_stream_positions(n, first_obs)

        preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions, group_size)
        horizon_preds = preds[:, ll * ps : n * ps]
        anchor_loc = loc[:, (ll - 1) * ps : (ll - 1) * ps + 1]
        anchor_scale = scale[:, (ll - 1) * ps : (ll - 1) * ps + 1]
        return denormalize_quantiles(horizon_preds, anchor_loc[..., None], anchor_scale[..., None], c.use_asinh)
