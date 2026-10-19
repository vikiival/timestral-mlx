"""Normalization for Timestral: causal per-patch mean/std scaling with an asinh squash.

Each patch is normalized with the mean and std of every observed value up to the end of
that patch, so the context never sees future statistics. Forecasts come back in the
normalized (asinh) space of the last context patch and are inverted with `sinh`.

The running statistics are float64 Welford states, as in training. Metal has no float64,
so those functions run on the CPU stream; Apple silicon's unified memory makes the hop
to and from the GPU free of copies.
"""

import mlx.core as mx
import numpy as np

FP32_EPS = float(np.finfo(np.float32).eps)
FP32_MAX = float(np.finfo(np.float32).max)
FP64_TINY = float(np.finfo(np.float64).tiny)


def compute_causal_statistics(data: mx.array, observation_mask: mx.array, minimum_scale: float) -> tuple[mx.array, mx.array]:
    """Causal running mean and std along the last axis (Welford form, in float64).

    Returns `(means, scales)`, both shaped like `data`; position `t` uses the observed
    values at positions `0..t` only. Unobserved positions get zero weight. The variance
    uses Bessel's correction and the std is clamped to `minimum_scale` after the sqrt.
    (`stabilize_with_global` is rejected by the config, so it is not implemented.)
    """
    with mx.stream(mx.cpu):
        x = data.astype(mx.float64)
        w = observation_mask.astype(mx.float64)
        weighted = w * x
        cum_w = mx.cumsum(w, axis=-1)
        cum_x = mx.cumsum(weighted, axis=-1)
        denominator = mx.maximum(cum_w, 1.0)
        means = cum_x / denominator
        shifted = mx.concatenate([mx.zeros_like(means[..., :1]), means[..., :-1]], axis=-1)
        increment = (x - shifted) * (x - means) * w
        m2 = mx.cumsum(increment, axis=-1)
        variance = m2 / mx.maximum(denominator - 1.0, 1.0)
        scales = mx.maximum(mx.sqrt(variance), minimum_scale)
        return means.astype(data.dtype), scales.astype(data.dtype)


def causal_patch_loc_scale(
    data: mx.array, observation_mask: mx.array, patch_size: int, minimum_scale: float
) -> tuple[mx.array, mx.array]:
    """Per-patch `(loc, scale)` broadcast back to every time step of the patch.

    Each patch uses the causal statistics at its last time step, i.e. everything
    observed up to the end of that patch.
    """
    assert data.shape[-1] % patch_size == 0, f"length {data.shape[-1]} is not a multiple of patch_size {patch_size}"
    means, scales = compute_causal_statistics(data, observation_mask, minimum_scale)
    last = slice(patch_size - 1, None, patch_size)
    return mx.repeat(means[..., last], patch_size, axis=-1), mx.repeat(scales[..., last], patch_size, axis=-1)


def apply_affine_asinh_masked(
    data: mx.array, loc: mx.array, scale: mx.array, observation_mask: mx.array, use_asinh: bool
) -> mx.array:
    """`asinh((data - loc) / scale)` on observed positions; unobserved positions become 0 (float32)."""
    scaled = (data.astype(mx.float32) - loc.astype(mx.float32)) / mx.maximum(scale.astype(mx.float32), FP32_EPS)
    scaled = mx.where(observation_mask, mx.arcsinh(scaled) if use_asinh else scaled, 0.0)
    return scaled


def invert_affine_asinh(x: mx.array, loc: mx.array, scale: mx.array, use_asinh: bool) -> mx.array:
    """Inverse of the forward transform: `sinh(x) * scale + loc` (float32)."""
    x = x.astype(mx.float32)
    if use_asinh:
        x = mx.sinh(x)
    return x * scale.astype(mx.float32) + loc.astype(mx.float32)


def clamp_nonfinite(vals: mx.array) -> mx.array:
    """Map +/-inf (a sinh overflow) to the nearest finite value along the last axis.

    A position whose whole last axis is non-finite becomes 0. NaN is left untouched so
    that real instability still surfaces downstream.
    """
    finite = mx.isfinite(vals)
    hi = mx.max(mx.where(finite, vals, -mx.inf), axis=-1, keepdims=True)
    lo = mx.min(mx.where(finite, vals, mx.inf), axis=-1, keepdims=True)
    clamped = mx.where(vals == mx.inf, hi, mx.where(vals == -mx.inf, lo, vals))
    return mx.where(mx.isinf(clamped), 0.0, clamped)


def denormalize_quantiles(preds: mx.array, loc: mx.array, scale: mx.array, use_asinh: bool) -> mx.array:
    """Invert the normalization and clamp infinities; monotone quantiles stay in order."""
    return clamp_nonfinite(invert_affine_asinh(preds, loc, scale, use_asinh))


def welford_combine(
    w_a: mx.array, mean_a: mx.array, m2_a: mx.array, w_b: mx.array, mean_b: mx.array, m2_b: mx.array
) -> tuple[mx.array, mx.array, mx.array]:
    """Merge two float64 Welford states `(weight, mean, m2)` element-wise; `(0, 0, 0)` is the identity.

    Call on the CPU stream.
    """
    w = w_a + w_b
    w_safe = mx.maximum(w, FP64_TINY)
    delta = mean_b - mean_a
    mean = mean_a + delta * w_b / w_safe
    m2 = m2_a + m2_b + mx.square(delta) * w_a * w_b / w_safe
    is_zero = w == 0
    return w, mx.where(is_zero, 0.0, mean), mx.where(is_zero, 0.0, m2)


def _shift_right(x: mx.array, d: int) -> mx.array:
    """Shift along axis 1 by `d`, filling with zeros."""
    return mx.concatenate([mx.zeros_like(x[:, :d]), x[:, : x.shape[1] - d]], axis=1)


def cumulative_welford(w: mx.array, mean: mx.array, m2: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Inclusive prefix scan of Welford states along axis 1 (Hillis-Steele, `log2(N)` merges)."""
    n = w.shape[1]
    d = 1
    while d < n:
        w, mean, m2 = welford_combine(_shift_right(w, d), _shift_right(mean, d), _shift_right(m2, d), w, mean, m2)
        d *= 2
    return w, mean, m2


def advancing_block_frames(
    context: mx.array, observation_mask: mx.array, sinh_median: mx.array, minimum_scale: float
) -> tuple[mx.array, mx.array]:
    """Per-patch `(loc, scale)` of a decoded block in the advancing refine frame, each `(R, H)` float64.

    Horizon patch `j` is normalized with the statistics of the observed context plus the
    block's medians of patches `0..j-1`, each median denormalized in its own frame before
    it is folded in. With perfect medians this is causal per-patch normalization of the true
    series. The statistics are float64 Welford states, accumulated exactly as in training.

    Args:
        context: `(R, T)` block context, `T` a multiple of the patch size.
        observation_mask: `(R, T)`.
        sinh_median: `(R, H, patch_size)` float64, the block's median in the last context
            patch's frame with the asinh undone, `(median - loc) / scale`.
        minimum_scale: Scale floor.
    """
    r, n_pred, ps = sinh_median.shape
    with mx.stream(mx.cpu):
        blank = mx.zeros((r, n_pred, ps), dtype=mx.float64)
        # Per-patch moments of the context, scanned together with the zero-weight horizon patches.
        values = mx.concatenate([context.reshape(r, -1, ps).astype(mx.float64), blank], axis=1)
        weights = mx.concatenate([observation_mask.reshape(r, -1, ps).astype(mx.float64), blank], axis=1)
        cnt = weights.sum(-1)
        mean = (values * weights).sum(-1) / mx.maximum(cnt, 1.0)
        m2 = (weights * mx.square(values - mean[..., None])).sum(-1)
        is_zero = cnt == 0
        mean = mx.where(is_zero, 0.0, mean)
        m2 = mx.where(is_zero, 0.0, m2)
        cnt, mean, m2 = (s[:, -n_pred - 1] for s in cumulative_welford(cnt, mean, m2))

        offset = mean  # medians are clamped relative to the context mean, so a large level survives
        clip = FP32_MAX**0.5 / 1e3
        locs, scales = [], []
        block_count = mx.full(cnt.shape, ps, dtype=mx.float64)
        for h in range(n_pred):
            var = mx.maximum(m2 / mx.maximum(cnt - 1.0, 1.0), 0.0)
            scale = mx.maximum(mx.sqrt(var), minimum_scale)
            locs.append(mean)
            scales.append(scale)
            folded = mx.clip(sinh_median[:, h] * scale[:, None] + mean[:, None] - offset[:, None], -clip, clip)
            folded = folded + offset[:, None]
            b_mean = folded.mean(-1)
            b_m2 = mx.square(folded - b_mean[:, None]).sum(-1)
            cnt, mean, m2 = welford_combine(cnt, mean, m2, block_count, b_mean, b_m2)
        return mx.stack(locs, axis=1), mx.stack(scales, axis=1)


def instance_norm_stats(
    values: mx.array, observation_mask: mx.array, minimum_scale: float
) -> tuple[mx.array, mx.array, mx.array]:
    """Non-causal per-row mean/std over observed positions, for known-future covariates.

    Rows whose observed values are all 0 or 1 (indicator covariates) keep `loc=0`,
    `scale=1`. Returns `(loc, scale, is_binary)`, each `(R, 1)`.
    """
    x = values.astype(mx.float32)
    mask_f = observation_mask.astype(mx.float32)
    n = mx.maximum(mask_f.sum(axis=-1, keepdims=True), 1.0)
    loc = (x * mask_f).sum(axis=-1, keepdims=True) / n
    var = mx.square((x - loc) * mask_f).sum(axis=-1, keepdims=True) / n
    scale = mx.maximum(mx.sqrt(var), minimum_scale)
    is_zero_one = ((values == 0.0) | (values == 1.0) | ~observation_mask).all(axis=-1, keepdims=True)
    is_binary = is_zero_one & observation_mask.any(axis=-1, keepdims=True)
    loc = mx.where(is_binary, 0.0, loc)
    scale = mx.where(is_binary, 1.0, scale)
    return loc.astype(values.dtype), scale.astype(values.dtype), is_binary
