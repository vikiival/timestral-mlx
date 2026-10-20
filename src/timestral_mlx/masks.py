"""Attention masks and time positions for Decoupled Horizon Forecasting (DHF) decoding.

The sequence is `[Enc ; Dec]`: encoder tokens embed the observed context patches;
decoder tokens are forecast queries for the horizon patches. All masks are boolean,
True meaning "may attend".

Rows come packed per series: series `b` owns rows `b * V .. b * V + V - 1`, `V` being the
group size. Variate masks are therefore built per series, `(B * N, 1, V, V)`.
"""

import mlx.core as mx


def compute_enc_key_validity(obs_per_patch: mx.array, attend_missing: bool) -> mx.array:
    """`(R, N)` encoder key validity.

    A patch is a valid key if it is observed, or if `attend_missing` is set and the
    row's series has started at or before it. Leading all-unobserved patches are
    left-padding and are never valid keys.
    """
    if not attend_missing:
        return obs_per_patch
    started = mx.cummax(obs_per_patch.astype(mx.int32), axis=-1).astype(mx.bool_)
    return obs_per_patch | started


def _tril(n: int) -> mx.array:
    return mx.tril(mx.ones((n, n), dtype=mx.bool_))


def _eye(n: int) -> mx.array:
    return mx.eye(n, dtype=mx.bool_)


def enc_causal_mask(key_valid: mx.array) -> mx.array:
    """`(R, L, L)` causal encoder mask over valid keys, with the diagonal always open."""
    _, ll = key_valid.shape
    return (_tril(ll)[None] & key_valid[:, None, :]) | _eye(ll)[None]


def build_forecast_time_mask(context_len: int, horizon: int, key_valid: mx.array) -> mx.array:
    """`(R, 1, L+h, L+h)` time mask for one block without covariates.

    The encoder is causal; every decoder token reads the whole valid context and its
    own causal decoder chain; the context never reads the horizon.
    """
    rows = key_valid.shape[0]
    enc = enc_causal_mask(key_valid)
    enc_dec = mx.zeros((rows, context_len, horizon), dtype=mx.bool_)
    dec_enc = mx.broadcast_to(key_valid[:, None, :], (rows, horizon, context_len))
    dec_dec = mx.broadcast_to(_tril(horizon)[None], (rows, horizon, horizon))
    top = mx.concatenate([enc, enc_dec], axis=2)
    bottom = mx.concatenate([dec_enc, dec_dec], axis=2)
    return mx.concatenate([top, bottom], axis=1)[:, None]


def build_forecast_positions(context_len: int, horizon: int, first_obs: mx.array) -> mx.array:
    """`(R, L+h)` time positions `0..L+h-1`, anchored at each row's first observed patch."""
    p = mx.arange(context_len + horizon)
    return p[None, :] - first_obs[:, None]


def build_window_time_mask(
    context_len: int, horizon: int, key_valid: mx.array, bidirectional_rows: mx.array | None
) -> mx.array:
    """`(R, 1, N+h, N+h)` time mask for one block with covariates, `N = L + h`.

    The encoder spans the context and the horizon, so known-future values are keys; the
    decoder holds the `h` horizon queries only.

    - Enc to Enc: causal over valid keys; full (non-causal) for `bidirectional_rows`,
      the known-future covariates.
    - Enc to Dec: never.
    - Dec to Enc: valid context keys `j < L`.
    - Dec to Dec: causal.

    The reference also builds decoder tokens for the `L` context patches. They only read
    themselves and nothing reads them, so leaving them out does not change the forecast.
    """
    rows, n = key_valid.shape
    eye = _eye(n)
    enc_enc = enc_causal_mask(key_valid)
    if bidirectional_rows is not None:
        enc_bi = mx.broadcast_to(key_valid[:, None, :], (rows, n, n)) | eye[None]
        enc_enc = mx.where(bidirectional_rows[:, None, None], enc_bi, enc_enc)
    enc_dec = mx.zeros((rows, n, horizon), dtype=mx.bool_)
    dec_enc = key_valid[:, None, :] & (mx.arange(n) < context_len)[None, None, :]
    dec_enc = mx.broadcast_to(dec_enc, (rows, horizon, n))
    dec_dec = mx.broadcast_to(_tril(horizon)[None], (rows, horizon, horizon))
    top = mx.concatenate([enc_enc, enc_dec], axis=2)
    bottom = mx.concatenate([dec_enc, dec_dec], axis=2)
    return mx.concatenate([top, bottom], axis=1)[:, None]


def variate_firewall(target_rows: mx.array | None, bidirectional_rows: mx.array | None) -> mx.array | None:
    """`(B, V, V)` forbidden query-to-key row pairs within each series, or None.

    Takes `(B, V)` row flags.

    - A covariate row may not read a target row (covariates are exogenous).
    - A known-future (bidirectional) row may not read a causal row. Without this, a
      covariate's view of the future could relay a target's own future back into it.
    """
    forbid = None
    if target_rows is not None:
        forbid = (~target_rows[:, :, None]) & target_rows[:, None, :]
    if bidirectional_rows is not None:
        bi = bidirectional_rows[:, :, None] & (~bidirectional_rows)[:, None, :]
        forbid = bi if forbid is None else forbid | bi
    return forbid


def build_two_stream_variate_mask(
    enc_key_valid: mx.array,
    n_dec: int,
    group_size: int,
    target_rows: mx.array | None = None,
    bidirectional_rows: mx.array | None = None,
) -> mx.array | None:
    """`(B * (N_enc + N_dec), 1, V, V)` variate mask, or None for one row per series.

    A row reads the other rows of its series; on encoder slots the key row must also be
    a valid key. The firewalls apply on every slot; the diagonal is always open. The
    leading axis is series-major, matching the `(B, N, V, D)` layout of variate attention.
    """
    if group_size == 1:
        return None
    r, n_enc = enc_key_valid.shape
    b, v = r // group_size, group_size
    slot_valid = mx.concatenate([enc_key_valid, mx.ones((r, n_dec), dtype=mx.bool_)], axis=1)
    n_slots = n_enc + n_dec
    key_valid = slot_valid.reshape(b, v, n_slots).transpose(0, 2, 1)  # (B, N, V): key row validity per slot
    mask = mx.broadcast_to(key_valid[:, :, None, :], (b, n_slots, v, v))
    grouped = lambda rows: rows.reshape(b, v) if rows is not None else None  # noqa: E731
    forbid = variate_firewall(grouped(target_rows), grouped(bidirectional_rows))
    if forbid is not None:
        mask = mask & ~forbid[:, None]
    mask = mask | _eye(v)
    return mask.reshape(b * n_slots, 1, v, v)


def build_window_positions(context_len: int, horizon: int, first_obs: mx.array) -> mx.array:
    """`(R, N+h)` positions: a decoder token shares the time of its horizon patch's encoder token."""
    row = mx.arange(context_len + horizon)[None, :] - first_obs[:, None]
    return mx.concatenate([row, row[:, context_len:]], axis=-1)


def group_first_observed_patch(obs_per_patch: mx.array, group_size: int) -> mx.array:
    """`(R,)` index of the first patch any row of the row's series observes (0 if none)."""
    r, n = obs_per_patch.shape
    group_obs = obs_per_patch.reshape(r // group_size, group_size, n).any(axis=1)
    first = mx.min(mx.where(group_obs, mx.arange(n), n), axis=-1)
    first = mx.where(first == n, 0, first)
    return mx.repeat(first, group_size, axis=0)
