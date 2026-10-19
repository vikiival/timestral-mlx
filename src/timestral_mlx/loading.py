"""Load exported checkpoints: a directory with `config.json` and `model.safetensors`, either on
disk or in a Hugging Face Hub repo."""

from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HFValidationError, RepositoryNotFoundError

from timestral_mlx.config import TimestralConfig
from timestral_mlx.model import TimestralModel

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


def resolve_checkpoint(
    path_or_repo_id: str | Path,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Return a local directory holding `config.json` and `model.safetensors`.

    An existing directory is used as is. Anything else is a Hugging Face Hub repo id
    (e.g. `mistralai/Timestral-5M`): its two files are downloaded to the Hub cache, or read
    from it when `local_files_only` (or `HF_HUB_OFFLINE=1`) is set.
    """
    path = Path(path_or_repo_id).expanduser()
    if path.is_dir():
        return path
    if path.exists():
        raise NotADirectoryError(
            f"{str(path)!r} is a file; pass the export directory holding {CONFIG_FILE} and {WEIGHTS_FILE}"
        )
    try:
        return Path(
            snapshot_download(
                str(path_or_repo_id),
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                allow_patterns=[CONFIG_FILE, WEIGHTS_FILE],
            )
        )
    except GatedRepoError:
        raise  # the Hub's message says how to request access
    except (RepositoryNotFoundError, HFValidationError) as e:
        raise FileNotFoundError(
            f"{str(path_or_repo_id)!r} is neither a local directory nor a Hugging Face Hub repo you can access"
        ) from e


def remap_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Rename released (PyTorch) parameter names to this package's module names.

    The attention gate is `_gate` in the weights; MLX does not treat underscore names as
    parameters, so the module calls it `gate`.
    """
    return {k.replace("._gate.", ".gate."): v for k, v in weights.items()}


def load_model(path_or_repo_id: str | Path, *, dtype: mx.Dtype = mx.float32, **hub_kwargs) -> TimestralModel:
    """Build the model from `config.json` and load `model.safetensors`.

    `path_or_repo_id` is a local directory or a Hugging Face Hub repo id; `hub_kwargs` go to
    `resolve_checkpoint`. Weights load strictly: every parameter must be present, with its
    shape. Inference was validated in float32.
    """
    path = resolve_checkpoint(path_or_repo_id, **hub_kwargs)
    config = TimestralConfig.load(path / CONFIG_FILE)
    model = TimestralModel(config)
    weights = remap_weights(mx.load(str(path / WEIGHTS_FILE)))
    model.load_weights([(k, v.astype(dtype)) for k, v in weights.items()], strict=True)
    mx.eval(model.parameters())
    return model.eval()
