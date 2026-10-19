"""Timestral time-series forecasting inference on Apple silicon with MLX."""

from timestral_mlx.config import TimestralConfig
from timestral_mlx.forecaster import Forecast, TimestralForecaster
from timestral_mlx.loading import load_model
from timestral_mlx.model import TimestralModel

__all__ = ["Forecast", "TimestralConfig", "TimestralForecaster", "TimestralModel", "load_model"]
__version__ = "0.1.0"
