"""Forecast precision against a float64 ground truth.

    uv run python benchmarks/precision.py --models 5M 209M

Both implementations compute in float32, so comparing MLX with PyTorch only shows how
far apart they are, not which one is closer to the exact answer. This script runs the
PyTorch reference in float64 as ground truth, in a subprocess whose float32 casts are
redirected to float64, then reports how far MLX and the float32 PyTorch reference each
are from it.

Error is the largest absolute difference divided by the largest absolute forecast value,
over all series, steps and quantiles: `1e-6` means one part in a million.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CASES = [  # (label, series, targets, past covariates, known covariates, context, horizon)
    ("16 x (1+k) 2048 -> 96", 16, 1, 0, 1, 2048, 96),
    ("8 x (3+p+k) 4096 -> 128", 8, 3, 1, 1, 4096, 128),
    ("4 x (1+k) 2000 -> 2500", 4, 1, 0, 1, 2000, 2500),
    ("1 x (1+k) 32768 -> 512", 1, 1, 0, 1, 32768, 512),
    ("32 x 1 2048 -> 96", 32, 1, 0, 0, 2048, 96),
    ("4 x 1 2000 -> 2500", 4, 1, 0, 0, 2000, 2500),
]


def make_inputs(b: int, v: int, p: int, k: int, t: int, h: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    season = 10 * np.sin(2 * np.pi * np.arange(t + h) / 24)
    x = (season[:t] + rng.standard_normal((b, v, t)).cumsum(-1)).astype(np.float32)
    covariates = {}
    if p:
        covariates["past_covariates"] = rng.standard_normal((b, p, t)).cumsum(-1).astype(np.float32)
    if k:
        covariates["known_covariates"] = (season + 0.1 * rng.standard_normal((b, k, t + h))).astype(np.float32)
    return x, covariates


def run_torch(repo: str, float64: bool) -> dict[str, np.ndarray]:
    import torch
    from timestral import TimestralForecaster as TorchForecaster

    forecaster = TorchForecaster.from_pretrained(repo)
    if float64:
        # The reference reads `torch.float32` at call time and casts with `.float()` in
        # RMSNorm and RoPE; point both at float64 so no step drops to float32. Only after
        # loading: safetensors reads `torch.float32` to size the stored weights.
        forecaster.model.double()
        torch.float32 = torch.float64
        torch.Tensor.float = torch.Tensor.double
    out = {}
    for label, *shape in CASES:
        x, covariates = make_inputs(*shape)
        kwargs = {name: torch.from_numpy(value) for name, value in covariates.items()}
        with torch.inference_mode():
            quantiles = forecaster.forecast(torch.from_numpy(x), shape[-1], **kwargs).quantiles
        assert quantiles.dtype == (torch.float64 if float64 else torch.float32), quantiles.dtype
        out[label] = quantiles.numpy()
    return out


def run_mlx(repo: str) -> dict[str, np.ndarray]:
    from timestral_mlx import TimestralForecaster

    forecaster = TimestralForecaster.from_pretrained(repo)
    out = {}
    for label, *shape in CASES:
        x, covariates = make_inputs(*shape)
        out[label] = forecaster.forecast(x, shape[-1], **covariates).quantiles
    return out


def error(actual: np.ndarray, truth: np.ndarray) -> float:
    truth = truth.astype(np.float64)
    return float(np.abs(actual.astype(np.float64) - truth).max() / max(1.0, np.abs(truth).max()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["5M", "209M"])
    parser.add_argument("--ground-truth", metavar="OUT", help=argparse.SUPPRESS)  # subprocess mode
    args = parser.parse_args()

    if args.ground_truth:
        np.savez(args.ground_truth, **run_torch(f"mistralai/Timestral-{args.models[0]}", float64=True))
        return

    print(f"{'model':>6} {'series x (targets+covariates) context -> horizon':>48} {'MLX':>9} {'PyTorch':>9}")
    for name in args.models:
        repo = f"mistralai/Timestral-{name}"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.npz"
            subprocess.run([sys.executable, __file__, "--models", name, "--ground-truth", str(path)], check=True)
            truth = dict(np.load(path))
        mlx_out = run_mlx(repo)
        torch_out = run_torch(repo, float64=False)
        for label, *_ in CASES:
            print(f"{name:>6} {label:>48} {error(mlx_out[label], truth[label]):9.1e} "
                  f"{error(torch_out[label], truth[label]):9.1e}", flush=True)


if __name__ == "__main__":
    main()
