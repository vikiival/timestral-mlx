# timestral-mlx

[Timestral](https://github.com/mistralai/timestral) time-series forecasting on Apple silicon, ported to [MLX](https://github.com/ml-explore/mlx).

Timestral is Mistral AI's family of time-series foundation models for multivariate probabilistic forecasting, with past-only and known-future covariates. The reference implementation is PyTorch. On a Mac it runs only on the CPU: its float64 normalization statistics fail on MPS. This port runs the network on the Apple GPU in float32 and keeps the float64 statistics on MLX's CPU stream. Unified memory means data doesn't move between the two.

The port matches the PyTorch reference to about 1e-6 relative difference on the released checkpoints (see [Parity](#parity)).

## Install

**To run forecasts only** (no PyTorch, no reference repo):

```bash
git clone <this repo> timestral-mlx && cd timestral-mlx
uv sync --frozen --no-dev    # mlx, numpy, huggingface_hub, as pinned in uv.lock
```

Pass `--frozen`. Without it, uv re-resolves the lockfile, which needs the reference repo to be present even when dev dependencies are skipped. `uv pip install .` also works.

**For development** (parity tests, benchmarks): the dev dependency group installs the PyTorch reference from the sibling directory `../timestral` (`[tool.uv.sources]` in `pyproject.toml`). Clone both repos side by side:

```bash
git clone https://github.com/mistralai/timestral.git
git clone <this repo> timestral-mlx
cd timestral-mlx && uv sync
```

The layout must be:

```
Playground/
├── timestral/        # PyTorch reference, unmodified
└── timestral-mlx/
```

## Forecast

The API matches the reference, except that results are NumPy arrays and there is no `device` argument.

```python
import numpy as np
from timestral_mlx import TimestralForecaster

forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")

context = np.random.randn(2, 1, 2000).cumsum(-1)        # (series, variates, time); NaN = missing
forecast = forecaster.forecast(context, prediction_length=96)

forecast.quantiles.shape   # (2, 1, 96, 21) float32 NumPy array: series, variates, horizon, quantile levels
forecast.median            # (2, 1, 96)
forecast.quantile(0.9)     # (2, 1, 96)
```

- `context` can be `(B, V, T)`, `(B, T)` or `(T,)`, as a NumPy array, an `mx.array` or anything `np.asarray` accepts.
- Covariates work as in the reference: `past_covariates`, `known_covariates`, `categorical_past_covariates` and `categorical_known_covariates`.
- `symmetric_averaging` and `clip_nonnegative` (clampsym) are on by default.

### Memory

Series are decoded in chunks to bound memory, using two limits:

- `max_batch_rows` (default 64): packed rows per model call. A packed row is one variate or covariate of one series.
- `max_batch_tokens` (default 65,536): packed rows × transformer tokens per model call. This drives peak memory. A row's tokens are its context patches plus one decode block, doubled when covariates are present.

With the defaults, Timestral-209M peaks at about 5 GB even at the maximum context (32,768 steps): 4.9 GB with 64 series, and 5.2 GB with 16 series plus a known covariate. Raise the limits for more throughput, or lower them on a smaller machine:

```python
forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", max_batch_tokens=32768)
```

A series is never split across chunks, so its variates are always forecast jointly. Options after the checkpoint path are keyword-only. The PyTorch reference's second positional argument is `device`, which doesn't exist here.

## Models

| model | Hub repo | weights (fp32) |
|---|---|---|
| Timestral-5M | [`mistralai/Timestral-5M`](https://huggingface.co/mistralai/Timestral-5M) | 21 MB |
| Timestral-28M | [`mistralai/Timestral-28M`](https://huggingface.co/mistralai/Timestral-28M) | 111 MB |
| Timestral-123M | [`mistralai/Timestral-123M`](https://huggingface.co/mistralai/Timestral-123M) | 490 MB |
| Timestral-209M | [`mistralai/Timestral-209M`](https://huggingface.co/mistralai/Timestral-209M) | 837 MB |

The released `model.safetensors` files load unchanged; the only key rename (`_gate` to `gate`) happens at load time.

## Differences from the reference

- **Variate attention is per series.** The reference builds one `(R, R)` attention over every row in the batch and masks out the other series. Here each series attends over its own `V` rows. That's the same result, but memory no longer grows with the square of the batch size. With one variate per series, the softmax weight is exactly 1, so variate attention returns `v` without running attention.
- **`predict_block` takes `num_series`, not `group_ids`.** Rows are always packed series by series with equal group sizes. Row roles are host-side NumPy arrays, so choosing a code path never waits on the GPU.
- **RoPE is hand-written.** Timestral pairs head dimension `i` with `i + head_dim / 2` and zero-pads the frequencies past `rotary_dim / 2`. `mx.fast.rope(dims=rotary_dim)` pairs `i` with `i + rotary_dim / 2`, which gives different results.
- **`stabilize_with_global` is not implemented.** The config already rejects it.

## Parity

`tests/test_parity.py` loads the same safetensors into both implementations. It compares full forecasts and the `predict_block` primitive across these cases:

- univariate and multivariate series
- multi-block horizons
- gaps, late starts and observation masks
- contexts beyond `max_context`
- large levels and magnitudes around 1e25
- constant series
- past, known, indicator and categorical covariates
- every clampsym combination
- the released Timestral-5M checkpoint

Measured on the released checkpoints (max abs diff / max |forecast|):

| checkpoint | 2000 → 96 | 1500 → 500 (16 blocks) | V=3 + known covariate |
|---|---|---|---|
| Timestral-5M | 5.4e-6 | 1.1e-6 | 3.4e-6 |
| Timestral-209M | 9.4e-7 | 1.1e-6 | 6.4e-7 |

```bash
uv run pytest                          # all tests; the 5M parity test downloads from the Hub
TIMESTRAL_SKIP_HUB=1 uv run pytest     # offline
```

## Benchmark

```bash
uv run python benchmarks/bench.py --models 5M 209M
```

On an M3 Pro (36 GB), MLX on the GPU vs the PyTorch reference on the CPU, median of 2 runs:

| model | series × variates × context → horizon | MLX | PyTorch CPU | speedup | MLX peak |
|---|---|---|---|---|---|
| 5M | 1 × 1 × 2048 → 96 | 14 ms | 10 ms | 0.7× | 0.04 GB |
| 5M | 32 × 1 × 2048 → 96 | 32 ms | 104 ms | 3.3× | 0.41 GB |
| 5M | 1 × 1 × 32768 → 512 | 25 ms | 77 ms | 3.1× | 0.21 GB |
| 209M | 1 × 1 × 2048 → 96 | 44 ms | 148 ms | 3.4× | 0.95 GB |
| 209M | 32 × 1 × 2048 → 96 | 0.64 s | 1.95 s | 3.0× | 2.69 GB |
| 209M | 256 × 1 × 512 → 64 | 1.40 s | 4.06 s | 2.9× | 2.36 GB |
| 209M | 8 × 4 × 4096 → 128 | 1.33 s | 4.16 s | 3.1× | 2.84 GB |
| 209M | 1 × 1 × 32768 → 512 | 0.38 s | 1.32 s | 3.5× | 2.33 GB |

A single short series on the 5M model is dominated by per-call overhead. The GPU only pays off once there's enough work.

## License

The code is Apache 2.0, derived from [mistralai/timestral](https://github.com/mistralai/timestral) (Apache 2.0). The Timestral weights are released separately on the Hugging Face Hub under a [Modified MIT license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE).
