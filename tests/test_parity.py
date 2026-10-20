"""Numerical parity with the PyTorch reference implementation (`../timestral`, a dev dependency).

Both sides load the same safetensors. The reference runs on CPU in float32 with float64
normalization statistics; the MLX port runs the network on the GPU in float32 and the
statistics on the CPU stream in float64. Differences come from float32 kernel ordering only.
"""

import math
import os

import mlx.core as mx
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from timestral import TimestralConfig as TorchConfig
from timestral import TimestralForecaster as TorchForecaster
from timestral import TimestralModel as TorchModel

from timestral_mlx import TimestralForecaster, load_model
from timestral_mlx.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET

KNOTS = (0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95)
RTOL = 1e-4


def assert_parity(actual: np.ndarray, expected: np.ndarray, rtol: float = RTOL) -> None:
    """Relative to the magnitude of the forecast, so tolerances do not depend on the series level."""
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    scale = max(1.0, float(np.abs(expected).max()))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * scale)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> tuple[TorchForecaster, TimestralForecaster]:
    """A random tiny checkpoint, saved once and loaded by both implementations."""
    config = TorchConfig(
        patch_size=8,
        d_model=32,
        num_heads=4,
        mlp_hidden_dim=64,
        block_schedule=("time", "time_variate", "time", "variate_time"),
        quantile_knots=KNOTS,
        block_size=4,
        context_length=256,
    )
    torch.manual_seed(0)
    model = TorchModel(config)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    path = tmp_path_factory.mktemp("tiny")
    config.save(path / "config.json")
    save_file(model.state_dict(), path / "model.safetensors")
    return TorchForecaster(model.eval()), TimestralForecaster(load_model(path))


def series(batch: int, variates: int, length: int, seed: int = 0, positive: bool = False, level: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)
    x = 10 * np.sin(2 * math.pi * t / 24) + rng.standard_normal((batch, variates, length)).cumsum(-1) + level
    return (np.abs(x) + 1 if positive else x).astype(np.float32)


def run_both(pair, x, horizon, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    ref, port = pair
    torch_kwargs = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
    expected = ref.forecast(torch.from_numpy(x), horizon, **torch_kwargs).quantiles.numpy()
    actual = port.forecast(x, horizon, **kwargs).quantiles
    return actual, expected


def test_weights_load_strictly(tiny) -> None:
    ref, port = tiny
    import mlx.utils

    n_ref = sum(p.numel() for p in ref.model.parameters())
    n_port = sum(v.size for _, v in mlx.utils.tree_flatten(port.model.parameters()))
    assert n_port == n_ref


@pytest.mark.parametrize(
    ("variates", "length", "horizon"),
    [(1, 100, 12), (3, 300, 50), (2, 37, 70), (1, 8, 8), (4, 64, 33)],
)
def test_forecast_parity(tiny, variates, length, horizon) -> None:
    x = series(3, variates, length, seed=variates + length)
    x[0, 0, 5:9] = np.nan
    actual, expected = run_both(tiny, x, horizon)
    assert np.isfinite(actual).all()
    assert (np.diff(actual, axis=-1) >= 0).all(), "quantiles must be sorted"
    assert_parity(actual, expected)


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("clip", [True, False])
def test_clampsym_options(tiny, symmetric, clip) -> None:
    x = series(2, 1, 120, seed=11, positive=True)
    actual, expected = run_both(tiny, x, 40, symmetric_averaging=symmetric, clip_nonnegative=clip)
    assert_parity(actual, expected)


def test_missing_values_and_late_start(tiny) -> None:
    """Leading padding (late start, negative positions), gaps, a whole-patch gap and an observation mask."""
    x = series(3, 2, 150, seed=5)
    x[0, :, :70] = np.nan  # series 0 starts late on every variate
    x[1, 0, :40] = np.nan  # one variate of series 1 starts late
    x[1, 1, 60:90] = np.nan
    x[2, 0, ::3] = np.nan
    mask = np.ones_like(x, dtype=bool)
    mask[2, 1, 100:110] = False
    actual, expected = run_both(tiny, x, 45, observation_mask=mask)
    assert_parity(actual, expected)


def test_context_beyond_max_context(tiny) -> None:
    x = series(2, 1, 700, seed=9)
    actual, expected = run_both(tiny, x, 100)
    assert_parity(actual, expected)


def test_large_level_and_extreme_magnitudes(tiny) -> None:
    x = series(2, 1, 120, seed=4, level=1e6)
    x[1] *= 1e25
    actual, expected = run_both(tiny, x, 30)
    assert_parity(actual, expected)


def test_constant_series(tiny) -> None:
    x = np.full((2, 1, 96), 3.0, dtype=np.float32)
    x[1] = 0.0
    actual, expected = run_both(tiny, x, 40)
    assert_parity(actual, expected)


def test_past_and_known_covariates(tiny) -> None:
    x = series(2, 1, 120, seed=3)
    past = series(2, 2, 120, seed=1)
    past[0, 1, 30:50] = np.nan
    known = series(2, 1, 120 + 50, seed=2)
    known[1, 0, 130:140] = np.nan
    actual, expected = run_both(tiny, x, 50, past_covariates=past, known_covariates=known)
    assert_parity(actual, expected)


def test_indicator_known_covariate_multivariate(tiny) -> None:
    x = series(2, 2, 96, seed=6)
    rng = np.random.default_rng(0)
    known = (rng.random((2, 2, 96 + 24)) > 0.5).astype(np.float32)
    actual, expected = run_both(tiny, x, 24, known_covariates=known)
    assert_parity(actual, expected)


def test_categorical_covariates(tiny) -> None:
    x = series(2, 1, 120, positive=True)
    rng = np.random.default_rng(7)
    cat_known = rng.integers(0, 4, (2, 1, 120 + 24)).astype(np.float64)
    cat_known[0, 0, 10:20] = np.nan
    cat_past = rng.integers(0, 3, (2, 1, 120)).astype(np.float64)
    actual, expected = run_both(
        tiny, x, 24, categorical_known_covariates=cat_known, categorical_past_covariates=cat_past
    )
    assert_parity(actual, expected)


def test_chunked_batch_matches_single_call(tiny) -> None:
    """Splitting the batch into chunks must not change any series' forecast."""
    _, port = tiny
    x = series(7, 2, 100, seed=12)
    full = port.forecast(x, 20).quantiles
    chunked = TimestralForecaster(port.model, max_batch_rows=4, max_batch_tokens=1000).forecast(x, 20).quantiles
    assert_parity(chunked, full, rtol=1e-5)


@pytest.mark.parametrize("h", [1, 3, 4])  # up to the tiny model's block_size
def test_predict_block_parity(tiny, h) -> None:
    """The network primitive alone, on both paths, before any forecaster logic."""
    ref, port = tiny
    rng = np.random.default_rng(1)
    b, v, t = 3, 3, 64
    x = rng.standard_normal((b * v, t)).astype(np.float32).cumsum(-1)
    obs = rng.random((b * v, t)) > 0.1
    obs[0, :20] = False
    x = np.where(obs, x, 0.0).astype(np.float32)
    group_ids = torch.arange(b).repeat_interleave(v)
    with torch.inference_mode():
        expected = ref.model.predict_block(torch.from_numpy(x), torch.from_numpy(obs), group_ids, h).numpy()
    actual = np.array(port.model.predict_block(mx.array(x), mx.array(obs), b, h))
    assert_parity(actual, expected)

    roles = np.tile([ROLE_TARGET, ROLE_PAST_COV, ROLE_KNOWN_COV], b)
    kf = rng.standard_normal((b * v, h * 8)).astype(np.float32)
    kf[roles != ROLE_KNOWN_COV] = 0.0
    kfm = np.broadcast_to((roles == ROLE_KNOWN_COV)[:, None], kf.shape).copy()
    with torch.inference_mode():
        expected = ref.model.predict_block(
            torch.from_numpy(x), torch.from_numpy(obs), group_ids, h,
            torch.from_numpy(roles), torch.from_numpy(kf), torch.from_numpy(kfm),
        ).numpy()
    actual = np.array(port.model.predict_block(mx.array(x), mx.array(obs), b, h, roles, mx.array(kf), mx.array(kfm)))
    assert_parity(actual, expected)


@pytest.mark.skipif(os.environ.get("TIMESTRAL_SKIP_HUB") == "1", reason="needs the Hugging Face Hub")
@pytest.mark.parametrize("repo", ["mistralai/Timestral-5M"])
def test_released_checkpoint_parity(repo) -> None:
    ref = TorchForecaster.from_pretrained(repo)
    port = TimestralForecaster.from_pretrained(repo)
    x = series(4, 1, 1500, seed=21)
    x[1, 0, :300] = np.nan
    actual, expected = run_both((ref, port), x, 200)
    assert_parity(actual, expected)

    xm = series(2, 3, 800, seed=22)
    known = series(2, 1, 800 + 64, seed=23)
    actual, expected = run_both((ref, port), xm, 64, known_covariates=known)
    assert_parity(actual, expected)
