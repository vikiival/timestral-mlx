"""Forecast precision against a float64 ground truth.

    uv run python benchmarks/precision.py --models 5M 209M

Both implementations compute in float32, so comparing MLX with PyTorch only shows how
far apart they are, not which one is closer to the exact answer. This script runs the
PyTorch reference in float64 as ground truth, in a subprocess whose float32 casts are
redirected to float64, then reports how far MLX and the float32 PyTorch reference each
are from it. The subprocess fails if any lossy float32 operation remains.

Error is the largest absolute difference divided by the largest absolute forecast value
(floored at 1, as in the parity tests), over all series, steps and quantiles: `1e-6`
means one part in a million.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CASES = [  # (series, targets, past covariates, known covariates, context, horizon)
    (16, 1, 0, 1, 2048, 96),
    (8, 3, 1, 1, 4096, 128),
    (4, 1, 0, 1, 2000, 2500),
    (1, 1, 0, 1, 32768, 512),
    (32, 1, 0, 0, 2048, 96),
    (4, 1, 0, 0, 2000, 2500),
]

# Operations that are exact on float32 data: indexing, copies, casts, sign flips and NaN
# handling. Anything else touching float32, or any op narrowing float64 data to float32,
# would round the ground truth.
EXACT_ON_FLOAT32 = {
    "_to_copy", "alias", "cat", "clone", "copy_", "detach", "expand", "isnan", "lift_fresh",
    "nan_to_num", "neg", "select", "slice", "to", "unsqueeze", "view",
}


def label(b: int, v: int, p: int, k: int, t: int, h: int) -> str:
    covariates = "+p" * bool(p) + "+k" * bool(k)
    return f"{b} x ({v}{covariates}) {t} -> {h}" if covariates else f"{b} x {v} {t} -> {h}"


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


def run_torch(forecaster) -> dict[str, np.ndarray]:
    import torch

    out = {}
    for shape in CASES:
        x, covariates = make_inputs(*shape)
        kwargs = {name: torch.from_numpy(value) for name, value in covariates.items()}
        with torch.inference_mode():
            out[label(*shape)] = forecaster.forecast(torch.from_numpy(x), shape[-1], **kwargs).quantiles.numpy()
    return out


def run_ground_truth(repo: str) -> dict[str, np.ndarray]:
    """The reference in float64. Patches torch for the rest of the process, so call it once."""
    import torch
    from timestral import TimestralForecaster as TorchForecaster
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    float32 = torch.float32
    assert float32 is not torch.float64, "torch is already patched; run the ground truth in a fresh process"
    forecaster = TorchForecaster.from_pretrained(repo)
    # The reference reads `torch.float32` at call time and casts with `.float()` in RMSNorm
    # and RoPE; point both at float64. Only after loading: safetensors reads `torch.float32`
    # to size the stored weights. The RoPE frequencies stay as built in float32, as in MLX
    # and in training: they are part of the model, not rounding in the forecast.
    forecaster.model.double()
    torch.float32 = torch.float64
    torch.Tensor.float = torch.Tensor.double

    class Float32Audit(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            ins = {t.dtype for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)}
            outs = {t.dtype for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)}
            if float32 in ins | outs:
                narrowing = float32 in outs and torch.float64 in ins
                exact = func.overloadpacket.__name__ in EXACT_ON_FLOAT32 and not narrowing
                assert exact, f"float32 {func} in the ground truth"
            return out

    with Float32Audit():
        out = run_torch(forecaster)
    for name, quantiles in out.items():
        assert quantiles.dtype == np.float64, f"{name}: ground truth is {quantiles.dtype}"
    return out


def run_mlx(repo: str) -> dict[str, np.ndarray]:
    from timestral_mlx import TimestralForecaster

    forecaster = TimestralForecaster.from_pretrained(repo)
    out = {}
    for shape in CASES:
        x, covariates = make_inputs(*shape)
        out[label(*shape)] = forecaster.forecast(x, shape[-1], **covariates).quantiles
    return out


def error(actual: np.ndarray, truth: np.ndarray) -> float:
    truth = truth.astype(np.float64)
    return float(np.abs(actual.astype(np.float64) - truth).max() / max(1.0, np.abs(truth).max()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["5M", "209M"])
    parser.add_argument("--ground-truth", nargs=2, metavar=("MODEL", "OUT"), help=argparse.SUPPRESS)  # subprocess
    args = parser.parse_args()

    if args.ground_truth:
        model, path = args.ground_truth
        np.savez(path, **run_ground_truth(f"mistralai/Timestral-{model}"))
        return

    from timestral import TimestralForecaster as TorchForecaster

    print(f"{'model':>6} {'series x (targets+covariates) context -> horizon':>48} {'MLX':>9} {'PyTorch':>9}")
    for name in args.models:
        repo = f"mistralai/Timestral-{name}"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.npz"
            subprocess.run([sys.executable, __file__, "--ground-truth", name, str(path)], check=True)
            truth = dict(np.load(path))
        mlx_out = run_mlx(repo)
        torch_out = run_torch(TorchForecaster.from_pretrained(repo))
        for shape in CASES:
            key = label(*shape)
            print(f"{name:>6} {key:>48} {error(mlx_out[key], truth[key]):9.1e} "
                  f"{error(torch_out[key], truth[key]):9.1e}", flush=True)


if __name__ == "__main__":
    main()
