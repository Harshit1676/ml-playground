import numpy as np

from app.ml.linear_regression import (
    NumericalDivergenceError,
    generate_synthetic_linear_dataset,
    train_linear_regression,
)
from app.schemas.linear_regression import (
    EffectiveDataset,
    LinearRegressionRequest,
    LinearRegressionResponse,
    Metrics,
    ModelParameters,
    PredictionValues,
    TrainingSummary,
)


def run_linear_regression_experiment(request: LinearRegressionRequest) -> LinearRegressionResponse:
    if request.dataset.kind == "points":
        x = np.asarray(request.dataset.x, dtype=np.float64)
        y = np.asarray(request.dataset.y, dtype=np.float64)
    else:
        dataset = request.dataset
        x, y = generate_synthetic_linear_dataset(
            n_samples=dataset.n_samples,
            slope=dataset.slope,
            intercept=dataset.intercept,
            noise_std=dataset.noise_std,
            x_min=dataset.x_min,
            x_max=dataset.x_max,
            seed=request.seed,
        )

    params = request.parameters
    result = train_linear_regression(
        x,
        y,
        initial_slope=params.initial_slope,
        initial_intercept=params.initial_intercept,
        learning_rate=params.learning_rate,
        steps=params.steps,
    )
    return LinearRegressionResponse(
        seed=request.seed,
        dataset=EffectiveDataset(x=x.tolist(), y=y.tolist()),
        parameters=params,
        model=ModelParameters(slope=result.slope, intercept=result.intercept),
        predictions=PredictionValues(y_hat=result.predictions.tolist()),
        metrics=Metrics(mse=result.mse),
        training=TrainingSummary(
            steps_completed=result.steps_completed,
            loss_history=list(result.loss_history),
        ),
    )


__all__ = ["NumericalDivergenceError", "run_linear_regression_experiment"]
