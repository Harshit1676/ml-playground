import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from app.main import create_app
from app.ml.linear_regression import NumericalDivergenceError, train_linear_regression
from app.schemas.linear_regression import LinearRegressionRequest
from app.services.linear_regression import run_linear_regression_experiment

client = TestClient(create_app())


def points_request(*, steps: int = 0, learning_rate: float = 0.1) -> dict:
    return {
        "dataset": {"kind": "points", "x": [-1.0, 0.0, 1.0], "y": [-1.0, 1.0, 3.0]},
        "parameters": {
            "initial_slope": 0.0,
            "initial_intercept": 0.0,
            "learning_rate": learning_rate,
            "steps": steps,
        },
        "seed": 42,
    }


def test_zero_steps_returns_initial_predictions_and_loss() -> None:
    response = run_linear_regression_experiment(LinearRegressionRequest.model_validate(points_request()))

    assert response.predictions.y_hat == [0.0, 0.0, 0.0]
    assert response.metrics.mse == pytest.approx(11 / 3)
    assert response.training.steps_completed == 0
    assert response.training.loss_history == pytest.approx([11 / 3])


def test_hand_calculated_gradient_and_simultaneous_one_step_update() -> None:
    # At the origin, residuals are [1, -1, -3], so d_slope=-8/3 and d_intercept=-2.
    result = train_linear_regression(
        [-1.0, 0.0, 1.0], [-1.0, 1.0, 3.0], initial_slope=0.0,
        initial_intercept=0.0, learning_rate=0.1, steps=1,
    )

    assert result.slope == pytest.approx(4 / 15)
    assert result.intercept == pytest.approx(0.2)
    assert result.mse == pytest.approx(2.64296296296)
    assert len(result.loss_history) == 2


def test_loss_history_has_initial_and_each_post_update_loss() -> None:
    result = train_linear_regression(
        [-1.0, 0.0, 1.0], [-1.0, 1.0, 3.0], initial_slope=0.0,
        initial_intercept=0.0, learning_rate=0.1, steps=4,
    )

    assert len(result.loss_history) == 5
    assert result.loss_history[0] == pytest.approx(11 / 3)
    assert result.loss_history[-1] == result.mse


def test_finite_increasing_loss_is_reported_without_relabeling_divergence() -> None:
    result = train_linear_regression(
        [100.0], [1.0], initial_slope=0.0,
        initial_intercept=0.0, learning_rate=1.0, steps=1,
    )

    assert result.loss_history[1] > result.loss_history[0]
    assert np.isfinite(result.loss_history[1])


def test_seeded_synthetic_dataset_is_reproducible() -> None:
    request = {
        "dataset": {
            "kind": "synthetic_linear", "n_samples": 20, "slope": 2.0,
            "intercept": 1.0, "noise_std": 0.2, "x_min": -5.0, "x_max": 5.0,
        },
        "parameters": {"initial_slope": 0.0, "initial_intercept": 0.0, "learning_rate": 0.01, "steps": 3},
        "seed": 123,
    }
    first = run_linear_regression_experiment(LinearRegressionRequest.model_validate(request))
    second = run_linear_regression_experiment(LinearRegressionRequest.model_validate(request))

    assert first.model_dump() == second.model_dump()


def test_different_seeds_generate_different_synthetic_data() -> None:
    body = {
        "dataset": {
            "kind": "synthetic_linear", "n_samples": 20, "slope": 2.0,
            "intercept": 1.0, "noise_std": 0.2, "x_min": -5.0, "x_max": 5.0,
        },
        "parameters": {"initial_slope": 0.0, "initial_intercept": 0.0, "learning_rate": 0.01, "steps": 0},
        "seed": 123,
    }
    first = run_linear_regression_experiment(LinearRegressionRequest.model_validate(body))
    body["seed"] = 124
    second = run_linear_regression_experiment(LinearRegressionRequest.model_validate(body))

    assert first.dataset.x != second.dataset.x
    assert first.dataset.y != second.dataset.y


def test_stateless_continuation_matches_two_updates() -> None:
    initial = run_linear_regression_experiment(LinearRegressionRequest.model_validate(points_request(steps=2)))
    first = run_linear_regression_experiment(LinearRegressionRequest.model_validate(points_request(steps=1)))
    continued_body = points_request(steps=1)
    continued_body["parameters"]["initial_slope"] = first.model.slope
    continued_body["parameters"]["initial_intercept"] = first.model.intercept
    continued = run_linear_regression_experiment(LinearRegressionRequest.model_validate(continued_body))

    assert continued.model.slope == pytest.approx(initial.model.slope)
    assert continued.model.intercept == pytest.approx(initial.model.intercept)
    assert continued.predictions.y_hat == pytest.approx(initial.predictions.y_hat)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body["dataset"].update(y=[1.0]),
        lambda body: body["dataset"].update(x=[float("inf")]),
        lambda body: body["dataset"].update(x=[1_000_001.0]),
        lambda body: body["parameters"].update(initial_intercept=float("nan")),
        lambda body: body["parameters"].update(learning_rate=0.0),
        lambda body: body["parameters"].update(learning_rate=1.01),
        lambda body: body["parameters"].update(steps=501),
        lambda body: body.update(seed=4_294_967_296),
        lambda body: body["dataset"].update(x=[0.0] * 1001, y=[0.0] * 1001),
    ],
)
def test_invalid_dimensions_and_numeric_values_are_rejected(mutate) -> None:
    body = points_request()
    mutate(body)
    response = client.post(
        "/api/v1/experiments/linear-regression",
        content=json.dumps(body, allow_nan=True),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_synthetic_dataset_constraints_are_validated() -> None:
    body = {
        "dataset": {
            "kind": "synthetic_linear", "n_samples": 1, "slope": 1.0,
            "intercept": 0.0, "noise_std": -0.1, "x_min": 2.0, "x_max": 1.0,
        },
        "parameters": {"initial_slope": 0.0, "initial_intercept": 0.0, "learning_rate": 0.1, "steps": 0},
        "seed": 0,
    }
    response = client.post("/api/v1/experiments/linear-regression", json=body)
    assert response.status_code == 422


def test_inline_observations_at_magnitude_boundary_are_accepted() -> None:
    body = points_request()
    body["dataset"] = {"kind": "points", "x": [-1_000_000.0, 1_000_000.0], "y": [-1_000_000.0, 1_000_000.0]}
    response = client.post("/api/v1/experiments/linear-regression", json=body)

    assert response.status_code == 200
    assert response.json()["dataset"]["x"] == [-1_000_000.0, 1_000_000.0]
    assert response.json()["dataset"]["y"] == [-1_000_000.0, 1_000_000.0]


@pytest.mark.parametrize(
    "dataset",
    [
        {"kind": "synthetic_linear", "n_samples": 2, "slope": 1_000_000.0, "intercept": -1_000_000.0, "noise_std": 0.0, "x_min": 0.0, "x_max": 1.0},
        {"kind": "synthetic_linear", "n_samples": 1000, "slope": 0.0, "intercept": 0.0, "noise_std": 1000.0, "x_min": -1_000_000.0, "x_max": 1_000_000.0},
    ],
)
def test_synthetic_parameter_boundaries_are_accepted_when_targets_fit(dataset: dict) -> None:
    body = {
        "dataset": dataset,
        "parameters": {"initial_slope": 0.0, "initial_intercept": 0.0, "learning_rate": 0.1, "steps": 0},
        "seed": 42,
    }
    response = client.post("/api/v1/experiments/linear-regression", json=body)

    assert response.status_code == 200
    assert len(response.json()["dataset"]["x"]) == dataset["n_samples"]


def test_synthetic_target_at_magnitude_boundary_is_accepted() -> None:
    body = {
        "dataset": {"kind": "synthetic_linear", "n_samples": 2, "slope": 0.0, "intercept": 1_000_000.0, "noise_std": 0.0, "x_min": -1_000_000.0, "x_max": 1_000_000.0},
        "parameters": {"initial_slope": 0.0, "initial_intercept": 0.0, "learning_rate": 0.1, "steps": 0},
        "seed": 42,
    }
    response = client.post("/api/v1/experiments/linear-regression", json=body)

    assert response.status_code == 200
    assert response.json()["dataset"]["y"] == [1_000_000.0, 1_000_000.0]


def test_generated_target_over_magnitude_boundary_is_structured_validation_error() -> None:
    body = {
        "dataset": {"kind": "synthetic_linear", "n_samples": 2, "slope": 1_000_000.0, "intercept": 0.0, "noise_std": 0.0, "x_min": 0.0, "x_max": 1_000_000.0},
        "parameters": {"initial_slope": 0.0, "initial_intercept": 0.0, "learning_rate": 0.1, "steps": 0},
        "seed": 42,
    }
    response = client.post("/api/v1/experiments/linear-regression", json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "DATASET_VALUE_OUT_OF_RANGE"
    assert set(response.json()["error"]) == {"code", "message", "details"}
    json.dumps(response.json(), allow_nan=False)
    assert response.json()["error"]["details"]["field"] == "dataset.y"
    assert response.json()["error"]["details"]["max_abs"] == 1_000_000.0
    assert response.json()["error"]["details"]["actual"] > 1_000_000.0
    assert "NUMERICAL_DIVERGENCE" not in response.text


def test_exactly_one_mib_request_body_is_accepted() -> None:
    body = json.dumps(points_request()).encode("utf-8")
    body += b" " * (1024 * 1024 - len(body))
    assert len(body) == 1024 * 1024

    response = client.post(
        "/api/v1/experiments/linear-regression",
        content=body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200


def test_oversized_request_returns_structured_413() -> None:
    response = client.post(
        "/api/v1/experiments/linear-regression",
        content=b" " * (1024 * 1024 + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_chunked_oversized_request_is_rejected_without_content_length() -> None:
    response = client.post(
        "/api/v1/experiments/linear-regression",
        content=iter([b" " * (1024 * 1024), b" "]),
        headers={"content-type": "application/json"},
    )

    assert "content-length" not in response.request.headers
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_malformed_json_returns_structured_validation_error() -> None:
    response = client.post(
        "/api/v1/experiments/linear-regression",
        content=b'{"dataset":',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["details"]["fields"]


def test_successful_response_contains_no_nan_or_infinity() -> None:
    response = client.post("/api/v1/experiments/linear-regression", json=points_request(steps=1))

    assert response.status_code == 200
    json.dumps(response.json(), allow_nan=False)
    assert "nan" not in response.text.lower()
    assert "infinity" not in response.text.lower()


def test_numerical_divergence_identifies_first_failing_intermediate() -> None:
    with pytest.raises(NumericalDivergenceError) as error:
        train_linear_regression(
            [1_000_000.0], [-1_000_000.0], initial_slope=1_000_000.0,
            initial_intercept=1_000_000.0, learning_rate=1.0, steps=20,
        )

    assert (error.value.step, error.value.value) == (12, "loss")


def test_numerical_divergence_api_is_structured_and_never_partial_success() -> None:
    body = {
        "dataset": {"kind": "points", "x": [1_000_000.0], "y": [-1_000_000.0]},
        "parameters": {
            "initial_slope": 1_000_000.0, "initial_intercept": 1_000_000.0,
            "learning_rate": 1.0, "steps": 20,
        },
        "seed": 42,
    }
    response = client.post("/api/v1/experiments/linear-regression", json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "NUMERICAL_DIVERGENCE"
    with pytest.raises(NumericalDivergenceError) as direct_error:
        train_linear_regression(
            [1_000_000.0], [-1_000_000.0], initial_slope=1_000_000.0,
            initial_intercept=1_000_000.0, learning_rate=1.0, steps=20,
        )
    assert response.json()["error"]["details"] == {
        "step": direct_error.value.step,
        "value": direct_error.value.value,
    }
    assert response.json()["error"]["details"] == {"step": 12, "value": "loss"}
    assert "model" not in response.json()
    assert "nan" not in response.text.lower()
    assert "infinity" not in response.text.lower()


def test_large_finite_intermediates_are_not_mistaken_for_divergence() -> None:
    result = train_linear_regression(
        [1_000_000.0], [-1_000_000.0], initial_slope=1_000_000.0,
        initial_intercept=1_000_000.0, learning_rate=1.0, steps=1,
    )
    assert np.isfinite(result.mse)
    assert result.steps_completed == 1


def test_converged_result_matches_scikit_learn_reference() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, size=100)
    y = 2.5 * x - 0.7 + rng.normal(0.0, 0.05, size=100)
    reference = LinearRegression().fit(x.reshape(-1, 1), y)
    result = train_linear_regression(
        x, y, initial_slope=0.0, initial_intercept=0.0, learning_rate=0.1, steps=500,
    )

    assert result.slope == pytest.approx(reference.coef_[0], abs=1e-8)
    assert result.intercept == pytest.approx(reference.intercept_, abs=1e-8)


def test_api_happy_path_returns_documented_contract() -> None:
    response = client.post("/api/v1/experiments/linear-regression", json=points_request(steps=1))

    assert response.status_code == 200
    body = response.json()
    assert body["algorithm"] == "linear_regression"
    assert body["training"]["steps_completed"] == 1
    assert len(body["training"]["loss_history"]) == 2
    assert body["metrics"]["mse"] == pytest.approx(body["training"]["loss_history"][-1])
    assert json.dumps(body, allow_nan=False)


def test_post_preflight_is_enabled_for_configured_local_origin() -> None:
    response = client.options(
        "/api/v1/experiments/linear-regression",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "POST" in response.headers["access-control-allow-methods"]
