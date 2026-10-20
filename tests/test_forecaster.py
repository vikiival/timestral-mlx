"""Behavioral tests of the MLX forecaster, independent of the PyTorch reference."""

import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

import timestral_mlx.loading
from timestral_mlx import Forecast, TimestralConfig, TimestralForecaster, TimestralModel
from timestral_mlx.covariate_encoding import target_encode

KNOTS = (0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95)


def tiny_forecaster(block_size: int = 4, max_context: int = 256) -> TimestralForecaster:
    config = TimestralConfig(
        patch_size=8,
        d_model=32,
        num_heads=4,
        mlp_hidden_dim=64,
        block_schedule=("time", "time_variate", "time", "time_variate"),
        quantile_knots=KNOTS,
        block_size=block_size,
        context_length=max_context,
    )
    mx.random.seed(0)
    model = TimestralModel(config)
    params = tree_flatten(model.parameters())
    model.load_weights([(k, 0.05 * mx.random.normal(v.shape)) for k, v in params], strict=True)
    return TimestralForecaster(model.eval())


def series(batch: int, variates: int, length: int, seed: int = 0, positive: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)
    x = 10 * np.sin(2 * math.pi * t / 24) + rng.standard_normal((batch, variates, length)).cumsum(-1)
    return (np.abs(x) + 1 if positive else x).astype(np.float32)


@pytest.mark.parametrize(
    ("variates", "length", "horizon"),
    [(1, 100, 12), (3, 300, 50), (2, 37, 70)],  # one block; context past the cap; three blocks after a short context
)
def test_forecast_shapes_and_finite(variates: int, length: int, horizon: int) -> None:
    fc = tiny_forecaster()
    x = series(2, variates, length)
    x[0, 0, 5:9] = np.nan
    out = fc.forecast(x, prediction_length=horizon)
    assert isinstance(out, Forecast)
    assert isinstance(out.quantiles, np.ndarray) and out.quantiles.dtype == np.float32
    assert out.quantiles.shape == (2, variates, horizon, len(KNOTS))
    assert np.isfinite(out.quantiles).all()
    assert (np.diff(out.quantiles, axis=-1) >= 0).all(), "quantiles must be sorted"
    assert out.median.shape == (2, variates, horizon)
    np.testing.assert_array_equal(out.quantile(0.9), out.quantiles[..., KNOTS.index(0.9)])


@pytest.mark.parametrize("shape", [(50,), (3, 50)])
def test_low_rank_contexts(shape) -> None:
    fc = tiny_forecaster()
    x = np.random.default_rng(0).standard_normal(shape).cumsum(-1)
    out = fc.forecast(x, 10)
    expected_b = 1 if len(shape) == 1 else shape[0]
    assert out.quantiles.shape == (expected_b, 1, 10, len(KNOTS))


def test_accepts_mlx_arrays() -> None:
    fc = tiny_forecaster()
    x = series(2, 1, 64)
    np.testing.assert_array_equal(fc.forecast(mx.array(x), 16).quantiles, fc.forecast(x, 16).quantiles)


def test_symmetric_averaging_is_odd_with_covariates() -> None:
    """The negated pass negates the covariates too, so negating every input reflects the forecast exactly."""
    fc = tiny_forecaster()
    x = series(2, 1, 120, seed=3)
    past = series(2, 2, 120, seed=1)
    known = series(2, 1, 120 + 24, seed=2)
    pos = fc.forecast(x, 24, past_covariates=past, known_covariates=known, clip_nonnegative=False).quantiles
    neg = fc.forecast(-x, 24, past_covariates=-past, known_covariates=-known, clip_nonnegative=False).quantiles
    np.testing.assert_array_equal(neg, -pos[..., ::-1])


def test_clip_nonnegative() -> None:
    fc = tiny_forecaster()
    x = series(2, 1, 100, positive=True)
    x[1, 0, 50] = -1.0  # series 1 is not eligible
    raw = fc.forecast(x, 24, clip_nonnegative=False).quantiles
    clipped = fc.forecast(x, 24).quantiles
    np.testing.assert_array_equal(clipped[0], np.maximum(raw[0], 0))
    np.testing.assert_array_equal(clipped[1], raw[1])


def test_categorical_covariates_are_target_encoded() -> None:
    fc = tiny_forecaster()
    x = series(2, 1, 120, positive=True)
    rng = np.random.default_rng(7)
    cat_known = rng.integers(0, 4, (2, 1, 120 + 24)).astype(np.float32)
    cat_known[0, 0, 10:20] = np.nan
    out = fc.forecast(x, 24, categorical_known_covariates=cat_known, symmetric_averaging=False).quantiles
    encoded = np.stack(
        [np.concatenate(target_encode(x[i, 0], cat_known[i, 0, :120], cat_known[i, 0, 120:])) for i in range(2)]
    )[:, None]
    np.testing.assert_allclose(out, fc.forecast(x, 24, known_covariates=encoded, symmetric_averaging=False).quantiles)


def test_rejects_bad_inputs() -> None:
    fc = tiny_forecaster()
    x = series(2, 1, 64)
    with pytest.raises(AssertionError):
        fc.forecast(x, 0)
    with pytest.raises(AssertionError):
        fc.forecast(x, 8, past_covariates=series(2, 1, 63))
    with pytest.raises(ValueError, match="infinite"):
        cats = np.zeros((2, 1, 64))
        cats[0, 0, 3] = np.inf
        fc.forecast(x, 8, categorical_past_covariates=cats)
    with pytest.raises(ValueError, match="not one of"):
        fc.forecast(x, 8).quantile(0.33)


def test_from_pretrained_local_directory_and_hub_repo(tmp_path, monkeypatch) -> None:
    fc = tiny_forecaster()
    fc.config.save(tmp_path / "config.json")
    weights = {k.replace(".gate.", "._gate."): v for k, v in tree_flatten(fc.model.parameters())}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    x = series(2, 1, 100)
    expected = fc.forecast(x, prediction_length=12).quantiles

    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(timestral_mlx.loading, "snapshot_download", fake_snapshot_download)
    local = TimestralForecaster.from_pretrained(tmp_path)
    assert not calls  # an existing directory never touches the Hub
    np.testing.assert_array_equal(local.forecast(x, prediction_length=12).quantiles, expected)

    hub = TimestralForecaster.from_pretrained("org/timestral-tiny", revision="v1", token=False, local_files_only=True)
    assert calls == [("org/timestral-tiny", {"revision": "v1", "cache_dir": None, "token": False,
                                             "local_files_only": True,
                                             "allow_patterns": ["config.json", "model.safetensors"]})]
    np.testing.assert_array_equal(hub.forecast(x, prediction_length=12).quantiles, expected)


def test_options_are_keyword_only() -> None:
    """The reference's second positional argument is `device`; it must not land in `max_context`."""
    fc = tiny_forecaster()
    with pytest.raises(TypeError):
        TimestralForecaster(fc.model, 128)
    with pytest.raises(TypeError):
        TimestralForecaster.from_pretrained("unused", "cpu")


def test_token_budget_bounds_chunks() -> None:
    fc = tiny_forecaster()
    budgeted = TimestralForecaster(fc.model, max_batch_tokens=1)
    # One series per chunk at the floor; context 256 / patch 8 = 32 patches, plus a 4-patch block.
    assert budgeted._series_per_chunk(2, 256, 24, windowed=False) == 1
    assert fc._series_per_chunk(1, 256, 24, windowed=False) == 64
    assert TimestralForecaster(fc.model, max_batch_tokens=36 * 10)._series_per_chunk(1, 256, 24, windowed=False) == 10
    # With covariates the encoder also spans the 4-patch block: 32 + 4 + 4 tokens per row.
    assert TimestralForecaster(fc.model, max_batch_tokens=40 * 10)._series_per_chunk(2, 256, 24, windowed=True) == 5
    x = series(5, 2, 100)
    np.testing.assert_allclose(budgeted.forecast(x, 20).quantiles, fc.forecast(x, 20).quantiles, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("rows", "tokens"), [(5, 65536), (64, 100), (1, 65536), (64, 1)])
def test_symmetric_averaging_is_odd_under_uneven_chunks(rows: int, tokens: int) -> None:
    """Chunking depends only on shapes, so negating the input still reflects the forecast exactly."""
    fc = TimestralForecaster(tiny_forecaster().model, max_batch_rows=rows, max_batch_tokens=tokens)
    x = series(3, 1, 120, seed=3)
    pos = fc.forecast(x, 40, clip_nonnegative=False).quantiles
    neg = fc.forecast(-x, 40, clip_nonnegative=False).quantiles
    np.testing.assert_array_equal(neg, -pos[..., ::-1])


def test_nan_error_counts_first_pass_only() -> None:
    fc = tiny_forecaster()
    fc.model.head.projection.output_layer.weight = mx.full(fc.model.head.projection.output_layer.weight.shape, mx.nan)
    with pytest.raises(ValueError, match=r"produced 72 NaN values"):  # 1 series x 8 steps x 9 quantiles, one pass
        fc.forecast(series(1, 1, 64), 8)
