# timestral-mlx

[Timestral](https://github.com/mistralai/timestral) time-series forecasting on Apple silicon, ported to [MLX](https://github.com/ml-explore/mlx).

Timestral is Mistral AI's family of time-series foundation models for multivariate probabilistic forecasting, with past-only and known-future covariates. The reference implementation is PyTorch. On a Mac it runs only on the CPU: its float64 normalization statistics fail on MPS. This port runs the network on the Apple GPU in float32 and keeps the float64 statistics on MLX's CPU stream. Unified memory means data doesn't move between the two.

The port matches the PyTorch reference to about 1e-6 relative difference on the released checkpoints (see [Parity](#parity)).

## Install

**To run forecasts only** (no PyTorch, no reference repo):

```bash
git clone https://github.com/vikiival/timestral-mlx.git && cd timestral-mlx
uv sync --frozen --no-dev    # mlx, numpy, huggingface_hub, as pinned in uv.lock
```

Pass `--frozen`. Without it, uv re-resolves the lockfile, which needs the reference repo to be present even when dev dependencies are skipped. `uv pip install .` also works.

**For development** (parity tests, benchmarks): the dev dependency group installs the PyTorch reference from the sibling directory `../timestral` (`[tool.uv.sources]` in `pyproject.toml`). Clone both repos side by side:

```bash
git clone https://github.com/mistralai/timestral.git
git clone https://github.com/vikiival/timestral-mlx.git
cd timestral-mlx && uv sync
```

The layout must be:

```
parent-dir/
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

Results are the same. Forecasts differ from PyTorch by about 1e-6 relative. Long multi-block horizons drift up to about 1e-4 on the 5M model, the same amount PyTorch drifts from itself (see [Parity](#parity)). Decoding batches in chunks changes memory use, not the results.

If you only call `forecast()`, you will notice three changes: the import name, NumPy output and the missing `device` argument.

### API

| | PyTorch reference | timestral-mlx |
|---|---|---|
| Import | `from timestral import ...` | `from timestral_mlx import ...` |
| Output | torch tensors | NumPy float32 arrays |
| Device | `device="cpu"` / `"cuda"`, `.device` property | none: GPU for the network, CPU stream for the float64 statistics |
| Input | NumPy arrays or torch tensors | NumPy arrays, `mx.array`, or anything `np.asarray` accepts |
| Options in `from_pretrained` and the constructor | by position or by name | by name only, after the path / model |
| Memory controls | none | `max_batch_rows`, `max_batch_tokens` |
| `load_model` | `(path, device, dtype)` | `(path, *, dtype)` |
| `predict_block` (low-level) | `group_ids`: any grouping of rows | `num_series`: series must be contiguous and equal-sized |

Options must be passed by name. Without a `device` argument, every later argument moved up one position, so a call copied from the reference would silently put `"cpu"` into `max_context`. Requiring names turns that into an immediate `TypeError`:

```python
# PyTorch reference: the 2nd argument is device, the 3rd is max_context
TimestralForecaster.from_pretrained("mistralai/Timestral-5M", "cpu", 4096)

# timestral-mlx: name the options
TimestralForecaster.from_pretrained("mistralai/Timestral-5M", max_context=4096)
TimestralForecaster.from_pretrained("mistralai/Timestral-5M", 4096)  # TypeError
```

The forecaster always packs rows as contiguous, equal-sized series, so the `predict_block` change only matters if you call that primitive directly. Row roles are host-side NumPy arrays, so picking a code path never waits on the GPU.

### Weights

The released `model.safetensors` files load unchanged. The attention gate is called `_gate` in the file and `gate` here, because MLX doesn't treat names that start with an underscore as parameters. It's renamed at load time. Weights saved from the MLX model need the reverse rename before PyTorch can load them.

### Platform and speed

The reference runs on CUDA GPUs or any CPU. On a Mac it's CPU-only, because MPS has no float64 and the normalization statistics need it. timestral-mlx targets Apple silicon, where it is about 3× faster than the reference on an M3 Pro. Timestral-209M peaks at about 5 GB of memory (see [Benchmark](#benchmark) and [Memory](#memory)).

### Not carried over

- `stabilize_with_global`: the reference config already rejects it, so nothing is lost.
- Low precision (bf16 / fp16): passing `dtype` to `load_model` works, but only float32 is tested, as in the reference.

### Internals (same math)

- Variate attention runs per series. The reference builds one `(R, R)` attention over every row in the batch and masks out the other series. Here each series attends over its own `V` rows, so memory no longer grows with the square of the batch size. With one variate per series the softmax weight is exactly 1, so variate attention returns `v` without running attention.
- RoPE is hand-written. Timestral pairs head dimension `i` with `i + head_dim / 2` and zero-pads the frequencies past `rotary_dim / 2`. `mx.fast.rope(dims=rotary_dim)` pairs `i` with `i + rotary_dim / 2`, which gives different results.
- float64 runs on the CPU stream. The Welford normalization statistics stay in float64 as in training. Metal has no float64, so they run on MLX's CPU stream, and unified memory means the data isn't copied between CPU and GPU.

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

The table shows how far the MLX forecasts are from the PyTorch forecasts on the released checkpoints. Each number is the largest disagreement divided by the largest forecast value, so `1e-6` means the two differ by at most one part in a million.

| checkpoint | 2000 steps → 96 (1 block) | 1500 → 500 (1 block) | 2000 → 2500 (3 blocks) | 3 variates + known covariate, 800 → 64 |
|---|---|---|---|---|
| Timestral-5M | 5.4e-6 | 1.1e-6 | 9.7e-5 | 3.4e-6 |
| Timestral-209M | 9.4e-7 | 1.1e-6 | 4.6e-6 | 6.4e-7 |

The model forecasts up to 1,024 steps (32 patches of 32) per block. Longer horizons are decoded block by block, and each block's median feeds back as context. Tiny float32 differences can compound across blocks, which is why the 3-block column is larger. That is the model's own sensitivity, not a porting error: nudging the input by one float32 rounding step moves PyTorch's own forecast by the same amount (7.4e-5 for Timestral-5M), and the per-block pattern matches MLX's.

For scale, a real porting bug (the wrong RoPE pairing) gives about 0.5.

```bash
uv run pytest                          # all tests; the 5M parity test downloads from the Hub
TIMESTRAL_SKIP_HUB=1 uv run pytest     # offline
```

## Benchmark

```bash
uv run python benchmarks/bench.py --models 5M 209M
```

On an M3 Pro (36 GB), MLX on the GPU vs the PyTorch reference on the CPU, median of 5 runs:

| model | series × variates × context → horizon | MLX | PyTorch CPU | speedup | MLX peak |
|---|---|---|---|---|---|
| 5M | 1 × 1 × 2048 → 96 | 4 ms | 10 ms | 2.3× | 0.05 GB |
| 5M | 32 × 1 × 2048 → 96 | 29 ms | 89 ms | 3.1× | 0.80 GB |
| 5M | 1 × 1 × 32768 → 512 | 20 ms | 67 ms | 3.3× | 0.41 GB |
| 209M | 1 × 1 × 2048 → 96 | 29 ms | 138 ms | 4.7× | 1.05 GB |
| 209M | 32 × 1 × 2048 → 96 | 0.61 s | 1.70 s | 2.8× | 2.74 GB |
| 209M | 256 × 1 × 512 → 64 | 1.34 s | 3.65 s | 2.7× | 2.36 GB |
| 209M | 8 × 4 × 4096 → 128 | 1.29 s | 3.59 s | 2.8× | 2.80 GB |
| 209M | 1 × 1 × 32768 → 512 | 0.36 s | 1.25 s | 3.5× | 2.63 GB |

Symmetric averaging (on by default) forecasts the series and its negation. Both passes run as one batch, so small forecasts pay the fixed per-call overhead only once.

## License

The code is Apache 2.0, derived from [mistralai/timestral](https://github.com/mistralai/timestral) (Apache 2.0). The Timestral weights are released separately on the Hugging Face Hub under a [Modified MIT license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE). It grants no rights to anyone whose company or employer has more than $20 million in monthly revenue.
