"""Forecast latency: MLX (Apple GPU) vs the PyTorch reference (CPU; it cannot run on MPS).

    uv run python benchmarks/bench.py --models 5M 209M
    uv run python benchmarks/bench.py --models 209M --no-torch    # MLX only

Each case is timed after one warm-up call; reported times are the median of `--repeats`
runs. MLX peak memory is the Metal allocator's high-water mark for the case.
"""

import argparse
import statistics
import time

import mlx.core as mx
import numpy as np

from timestral_mlx import TimestralForecaster

CASES = [  # (series, variates, context, horizon)
    (1, 1, 2048, 96),
    (32, 1, 2048, 96),
    (256, 1, 512, 64),
    (8, 4, 4096, 128),
    (1, 1, 32768, 512),
]


def series(b: int, v: int, t: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (10 * np.sin(np.arange(t) * 2 * np.pi / 24) + rng.standard_normal((b, v, t)).cumsum(-1)).astype(np.float32)


def timed(fn, repeats: int) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["5M", "209M"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-torch", action="store_true")
    parser.add_argument("--max-batch-rows", type=int, default=64)
    args = parser.parse_args()

    if not args.no_torch:
        import torch
        from timestral import TimestralForecaster as TorchForecaster

    print(f"{'model':>6} {'B x V x T -> H':>22} {'mlx s':>8} {'peak GB':>8} {'torch s':>8} {'speedup':>8}")
    for name in args.models:
        repo = f"mistralai/Timestral-{name}"
        port = TimestralForecaster.from_pretrained(repo, max_batch_rows=args.max_batch_rows)
        ref = None if args.no_torch else TorchForecaster.from_pretrained(repo)
        for b, v, t, h in CASES:
            x = series(b, v, t)
            mx.reset_peak_memory()
            t_mlx = timed(lambda: port.forecast(x, h), args.repeats)
            peak = mx.get_peak_memory() / 1e9
            row = f"{name:>6} {f'{b} x {v} x {t} -> {h}':>22} {t_mlx:8.3f} {peak:8.2f}"
            if ref is not None:
                xt = torch.from_numpy(x)
                t_torch = timed(lambda: ref.forecast(xt, h), args.repeats)
                row += f" {t_torch:8.3f} {t_torch / t_mlx:7.1f}x"
            print(row, flush=True)


if __name__ == "__main__":
    main()
