"""NumPy implementation of the educational batch gradient-descent engine."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrainingResult:
    slope: float
    intercept: float
    predictions: FloatArray
    mse: float
    loss_history: tuple[float, ...]
    steps_completed: int


class NumericalDivergenceError(Exception):
    """Raised when an intermediate becomes non-finite, identifying its first step."""

    def __init__(self, step: int, value: str) -> None:
        super().__init__(f"Non-finite value in {value} at step {step}.")
        self.step = step
        self.value = value


def _require_finite(value: FloatArray | np.float64 | float, *, step: int, field: str) -> None:
    if not bool(np.all(np.isfinite(value))):
        raise NumericalDivergenceError(step, field)


def train_linear_regression(
    x_values: list[float] | FloatArray,
    y_values: list[float] | FloatArray,
    *,
    initial_slope: float,
    initial_intercept: float,
    learning_rate: float,
    steps: int,
) -> TrainingResult:
    """Train with simultaneous batch updates and return loss before/after every step."""
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size == 0:
        raise ValueError("x and y must be non-empty one-dimensional arrays of equal length")

    slope = np.float64(initial_slope)
    intercept = np.float64(initial_intercept)
    losses: list[float] = []

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        predictions = slope * x + intercept
        _require_finite(predictions, step=0, field="predictions")
        residuals = predictions - y
        _require_finite(residuals, step=0, field="residuals")
        loss = np.mean(residuals**2, dtype=np.float64)
        _require_finite(loss, step=0, field="loss")
        losses.append(float(loss))

        for step in range(1, steps + 1):
            gradient_slope = np.float64(2.0) * np.mean(residuals * x, dtype=np.float64)
            _require_finite(gradient_slope, step=step, field="gradient_slope")
            gradient_intercept = np.float64(2.0) * np.mean(residuals, dtype=np.float64)
            _require_finite(gradient_intercept, step=step, field="gradient_intercept")

            # Calculate both candidates from gradients of the same pre-update state.
            next_slope = slope - np.float64(learning_rate) * gradient_slope
            _require_finite(next_slope, step=step, field="candidate_slope")
            next_intercept = intercept - np.float64(learning_rate) * gradient_intercept
            _require_finite(next_intercept, step=step, field="candidate_intercept")

            next_predictions = next_slope * x + next_intercept
            _require_finite(next_predictions, step=step, field="predictions")
            next_residuals = next_predictions - y
            _require_finite(next_residuals, step=step, field="residuals")
            next_loss = np.mean(next_residuals**2, dtype=np.float64)
            _require_finite(next_loss, step=step, field="loss")

            slope, intercept = next_slope, next_intercept
            predictions, residuals, loss = next_predictions, next_residuals, next_loss
            losses.append(float(loss))

    return TrainingResult(
        slope=float(slope),
        intercept=float(intercept),
        predictions=predictions,
        mse=float(loss),
        loss_history=tuple(losses),
        steps_completed=steps,
    )


def generate_synthetic_linear_dataset(
    *,
    n_samples: int,
    slope: float,
    intercept: float,
    noise_std: float,
    x_min: float,
    x_max: float,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Generate repeatable x values and Gaussian-noise-contaminated y values."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n_samples).astype(np.float64, copy=False)
    noise = rng.normal(0.0, noise_std, size=n_samples).astype(np.float64, copy=False)
    with np.errstate(over="ignore", invalid="ignore"):
        y = np.float64(slope) * x + np.float64(intercept) + noise
    _require_finite(x, step=0, field="dataset_x")
    _require_finite(y, step=0, field="dataset_y")
    return x, y
