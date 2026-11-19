"""NumPy float64 engine for univariate binary Logistic Regression."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LogisticTrainingResult:
    coefficient: float
    intercept: float
    probabilities: FloatArray
    binary_cross_entropy: float
    loss_history: tuple[float, ...]
    steps_completed: int


class LogisticNumericalDivergenceError(Exception):
    """Raised when an intermediate becomes non-finite, identifying its step."""

    def __init__(self, step: int, value: str) -> None:
        super().__init__(f"Non-finite value in {value} at step {step}.")
        self.step = step
        self.value = value


class SingleClassSyntheticDatasetError(Exception):
    """Raised when seeded probabilistic generation produces one class only."""

    def __init__(self, class_label: int, n_samples: int) -> None:
        super().__init__(f"Seeded synthetic data produced only class {class_label}.")
        self.class_label = class_label
        self.n_samples = n_samples


def _require_finite(value: FloatArray | np.float64 | float, *, step: int, field: str) -> None:
    if not bool(np.all(np.isfinite(value))):
        raise LogisticNumericalDivergenceError(step, field)


def _finite_mean(values: FloatArray, *, step: int, field: str) -> np.float64:
    """Average finite values without overflowing an otherwise representable sum."""
    _require_finite(values, step=step, field=field)
    scale = np.max(np.abs(values))
    if scale == 0.0:
        return np.float64(0.0)

    normalized = values / scale
    _require_finite(normalized, step=step, field=field)
    normalized_mean = np.mean(normalized, dtype=np.float64)
    _require_finite(normalized_mean, step=step, field=field)
    mean = scale * normalized_mean
    _require_finite(mean, step=step, field=field)
    return np.float64(mean)


def stable_sigmoid(logits: FloatArray | list[float]) -> FloatArray:
    """Evaluate sigmoid without exponentiating a large positive logit."""
    values = np.asarray(logits, dtype=np.float64)
    _require_finite(values, step=0, field="logits")

    flat = values.reshape(-1)
    result = np.empty_like(flat)
    nonnegative = flat >= 0.0
    negative = ~nonnegative
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result[nonnegative] = 1.0 / (1.0 + np.exp(-flat[nonnegative]))
        exponentials = np.exp(flat[negative])
        result[negative] = exponentials / (1.0 + exponentials)
    _require_finite(result, step=0, field="probabilities")
    return result.reshape(values.shape)


def binary_cross_entropy_from_logits(
    logits: FloatArray | list[float],
    labels: FloatArray | list[float],
    *,
    step: int = 0,
) -> float:
    """Compute mean BCE using the stable softplus/logits identity."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if z.ndim != 1 or y.ndim != 1 or z.size == 0 or z.size != y.size:
        raise ValueError("logits and labels must be non-empty one-dimensional arrays of equal length")
    _require_finite(z, step=step, field="logits")
    _require_finite(y, step=step, field="labels")
    if not bool(np.all((y == 0.0) | (y == 1.0))):
        raise ValueError("labels must be exactly 0 or 1")

    with np.errstate(over="ignore", invalid="ignore"):
        observation_losses = np.logaddexp(0.0, z) - y * z
        _require_finite(observation_losses, step=step, field="loss")
        loss = _finite_mean(observation_losses, step=step, field="loss")
    _require_finite(loss, step=step, field="loss")
    return float(loss)


def train_logistic_regression(
    x_values: list[float] | FloatArray,
    y_values: list[float] | FloatArray,
    *,
    initial_coefficient: float,
    initial_intercept: float,
    learning_rate: float,
    steps: int,
) -> LogisticTrainingResult:
    """Train using simultaneous batch updates and return every BCE value."""
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0 or x.size != y.size:
        raise ValueError("x and y must be non-empty one-dimensional arrays of equal length")
    _require_finite(x, step=0, field="features")
    _require_finite(y, step=0, field="labels")
    if not bool(np.all((y == 0.0) | (y == 1.0))):
        raise ValueError("labels must be exactly 0 or 1")

    coefficient = np.float64(initial_coefficient)
    intercept = np.float64(initial_intercept)
    _require_finite(coefficient, step=0, field="coefficient")
    _require_finite(intercept, step=0, field="intercept")
    losses: list[float] = []

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logits = coefficient * x + intercept
        _require_finite(logits, step=0, field="logits")
        probabilities = stable_sigmoid(logits)
        residuals = probabilities - y
        _require_finite(residuals, step=0, field="residuals")
        loss = binary_cross_entropy_from_logits(logits, y)
        losses.append(loss)

        for step in range(1, steps + 1):
            coefficient_terms = residuals * x
            gradient_coefficient = _finite_mean(coefficient_terms, step=step, field="gradient_coefficient")
            _require_finite(gradient_coefficient, step=step, field="gradient_coefficient")
            gradient_intercept = _finite_mean(residuals, step=step, field="gradient_intercept")
            _require_finite(gradient_intercept, step=step, field="gradient_intercept")

            # Both candidates use gradients from the same pre-update parameters.
            next_coefficient = coefficient - np.float64(learning_rate) * gradient_coefficient
            _require_finite(next_coefficient, step=step, field="candidate_coefficient")
            next_intercept = intercept - np.float64(learning_rate) * gradient_intercept
            _require_finite(next_intercept, step=step, field="candidate_intercept")

            next_logits = next_coefficient * x + next_intercept
            _require_finite(next_logits, step=step, field="logits")
            next_probabilities = stable_sigmoid(next_logits)
            next_residuals = next_probabilities - y
            _require_finite(next_residuals, step=step, field="residuals")
            next_loss = binary_cross_entropy_from_logits(next_logits, y, step=step)

            coefficient, intercept = next_coefficient, next_intercept
            logits, probabilities, residuals, loss = next_logits, next_probabilities, next_residuals, next_loss
            losses.append(loss)

    return LogisticTrainingResult(
        coefficient=float(coefficient),
        intercept=float(intercept),
        probabilities=probabilities,
        binary_cross_entropy=float(loss),
        loss_history=tuple(losses),
        steps_completed=steps,
    )


def generate_synthetic_logistic_dataset(
    *,
    n_samples: int,
    true_coefficient: float,
    true_intercept: float,
    x_min: float,
    x_max: float,
    seed: int,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Sample features uniformly, then sample Bernoulli labels from sigmoid(z)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n_samples).astype(np.float64, copy=False)
    with np.errstate(over="ignore", invalid="ignore"):
        logits = np.float64(true_coefficient) * x + np.float64(true_intercept)
    _require_finite(x, step=0, field="dataset_x")
    _require_finite(logits, step=0, field="dataset_logits")
    probabilities = stable_sigmoid(logits)
    labels = rng.binomial(1, probabilities).astype(np.int64, copy=False)
    unique_labels = np.unique(labels)
    if unique_labels.size != 2:
        raise SingleClassSyntheticDatasetError(int(unique_labels[0]), n_samples)
    return x, labels
