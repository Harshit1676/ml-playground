"""Stateless Logistic Regression experiment orchestration and metrics."""

import numpy as np

from app.ml.logistic_regression import (
    LogisticNumericalDivergenceError,
    generate_synthetic_logistic_dataset,
    train_logistic_regression,
)
from app.schemas.logistic_regression import (
    ConfusionMatrix,
    EffectiveDataset,
    LogisticRegressionRequest,
    LogisticRegressionResponse,
    Metrics,
    ModelParameters,
    Predictions,
    TrainingSummary,
)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def run_logistic_regression_experiment(request: LogisticRegressionRequest) -> LogisticRegressionResponse:
    if request.dataset.kind == "points":
        x = np.asarray(request.dataset.x, dtype=np.float64)
        y = np.asarray(request.dataset.y, dtype=np.float64)
    else:
        dataset = request.dataset
        x, generated_y = generate_synthetic_logistic_dataset(
            n_samples=dataset.n_samples,
            true_coefficient=dataset.true_coefficient,
            true_intercept=dataset.true_intercept,
            x_min=dataset.x_min,
            x_max=dataset.x_max,
            seed=request.seed,
        )
        y = generated_y.astype(np.float64, copy=False)

    parameters = request.parameters
    result = train_logistic_regression(
        x,
        y,
        initial_coefficient=parameters.initial_coefficient,
        initial_intercept=parameters.initial_intercept,
        learning_rate=parameters.learning_rate,
        steps=parameters.steps,
    )

    predicted = result.probabilities >= request.threshold
    actual = y == 1.0
    true_positive = int(np.count_nonzero(predicted & actual))
    true_negative = int(np.count_nonzero(~predicted & ~actual))
    false_positive = int(np.count_nonzero(predicted & ~actual))
    false_negative = int(np.count_nonzero(~predicted & actual))
    accuracy = _safe_ratio(true_positive + true_negative, y.size)
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)

    return LogisticRegressionResponse(
        seed=request.seed,
        dataset=EffectiveDataset(x=x.tolist(), y=y.astype(np.int64).tolist()),
        parameters=parameters,
        threshold=request.threshold,
        model=ModelParameters(coefficient=result.coefficient, intercept=result.intercept),
        predictions=Predictions(
            probabilities=result.probabilities.tolist(),
            predicted_classes=predicted.astype(np.int64).tolist(),
        ),
        metrics=Metrics(
            binary_cross_entropy=result.binary_cross_entropy,
            confusion_matrix=ConfusionMatrix(
                true_positive=true_positive,
                true_negative=true_negative,
                false_positive=false_positive,
                false_negative=false_negative,
            ),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
        ),
        training=TrainingSummary(
            steps_completed=result.steps_completed,
            loss_history=list(result.loss_history),
        ),
    )


__all__ = ["LogisticNumericalDivergenceError", "run_logistic_regression_experiment"]
