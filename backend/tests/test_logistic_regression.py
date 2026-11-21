import json
import math

import numpy as np
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.api.routes import logistic_regression as logistic_route
from app.schemas.logistic_regression import LogisticRegressionRequest
from app.main import create_app
from app.ml.logistic_regression import (
    LogisticNumericalDivergenceError,
    binary_cross_entropy_from_logits,
    stable_sigmoid,
    train_logistic_regression,
)

client = TestClient(create_app())


def points_request(
    *,
    x: list[float] | None = None,
    y: list[int] | None = None,
    threshold: float = 0.5,
    steps: int = 0,
    learning_rate: float = 0.1,
) -> dict:
    return {
        "dataset": {
            "kind": "points",
            "x": [-1.0, 0.0, 1.0, 2.0] if x is None else x,
            "y": [0, 0, 1, 1] if y is None else y,
        },
        "parameters": {
            "initial_coefficient": 0.0,
            "initial_intercept": 0.0,
            "learning_rate": learning_rate,
            "steps": steps,
        },
        "threshold": threshold,
        "seed": 42,
    }


def default_synthetic_request() -> dict:
    return {
        "dataset": {
            "kind": "synthetic_logistic",
            "n_samples": 100,
            "true_coefficient": 1.5,
            "true_intercept": 0.0,
            "x_min": -4.0,
            "x_max": 4.0,
        },
        "parameters": {
            "initial_coefficient": 0.0,
            "initial_intercept": 0.0,
            "learning_rate": 0.1,
            "steps": 25,
        },
        "threshold": 0.5,
        "seed": 42,
    }


def test_stable_sigmoid_handles_negative_zero_and_positive_logits() -> None:
    probabilities = stable_sigmoid([-1000.0, 0.0, 1000.0])

    assert probabilities.tolist() == [0.0, 0.5, 1.0]
    assert np.isfinite(probabilities).all()


def test_logits_based_binary_cross_entropy_is_stable_for_extreme_logits() -> None:
    loss = binary_cross_entropy_from_logits([1000.0, -1000.0], [0.0, 1.0])

    assert loss == pytest.approx(1000.0)
    assert math.isfinite(loss)


def test_bce_mean_avoids_overflow_for_large_finite_loss_terms() -> None:
    loss = binary_cross_entropy_from_logits([1e308] * 4, [0.0] * 4)

    assert loss == pytest.approx(1e308)
    assert math.isfinite(loss)


def test_one_step_gradient_descent_matches_independent_fixture() -> None:
    result = train_logistic_regression(
        [-2.0, 0.0, 1.0, 3.0], [0.0, 1.0, 0.0, 1.0],
        initial_coefficient=0.3, initial_intercept=-0.2,
        learning_rate=0.15, steps=1,
    )

    # Independent scalar calculation: grad_w=-0.27262713344033418 and
    # grad_b=-0.011660379698246057, both from the initial logits.
    assert result.coefficient == pytest.approx(0.3408940700160501, abs=1e-14)
    assert result.intercept == pytest.approx(-0.19825094304526311, abs=1e-14)
    assert result.loss_history == pytest.approx(
        [0.57920556107209964, 0.56868554088904799], abs=1e-14
    )
    assert result.binary_cross_entropy == pytest.approx(0.56868554088904799, abs=1e-14)
    assert result.steps_completed == 1


def test_zero_steps_preserve_initial_parameters_and_record_only_initial_loss() -> None:
    result = train_logistic_regression(
        [-1.0, 2.0], [0.0, 1.0], initial_coefficient=0.75,
        initial_intercept=-0.25, learning_rate=0.2, steps=0,
    )

    assert result.coefficient == 0.75
    assert result.intercept == -0.25
    assert result.steps_completed == 0
    assert len(result.loss_history) == 1
    assert result.loss_history[0] == result.binary_cross_entropy


def test_one_step_updates_both_parameters_from_the_same_pre_update_state() -> None:
    result = train_logistic_regression(
        [0.0, 2.0], [0.0, 0.0], initial_coefficient=0.0,
        initial_intercept=0.0, learning_rate=0.1, steps=1,
    )

    # At p=[0.5, 0.5], both initial gradients are 0.5; sequential updates
    # would compute a different intercept gradient after changing the slope.
    assert result.coefficient == pytest.approx(-0.05)
    assert result.intercept == pytest.approx(-0.05)


def test_loss_decreases_on_a_controlled_separable_dataset() -> None:
    result = train_logistic_regression(
        [-2.0, -1.0, 1.0, 2.0], [0.0, 0.0, 1.0, 1.0],
        initial_coefficient=0.0, initial_intercept=0.0,
        learning_rate=0.2, steps=100,
    )

    assert result.loss_history[-1] < result.loss_history[0]
    assert len(result.loss_history) == 101


def test_engine_probabilities_match_final_model() -> None:
    x = np.asarray([-1.0, 0.0, 1.0])
    result = train_logistic_regression(
        x, [0.0, 0.0, 1.0], initial_coefficient=0.4,
        initial_intercept=-0.1, learning_rate=0.05, steps=4,
    )

    expected = stable_sigmoid(result.coefficient * x + result.intercept)
    assert result.probabilities == pytest.approx(expected)


def test_threshold_boundary_changes_classes_but_not_training_values() -> None:
    at_boundary = client.post("/api/v1/experiments/logistic-regression", json=points_request(x=[0.0, 1.0], y=[0, 1], threshold=0.5)).json()
    above_boundary = client.post("/api/v1/experiments/logistic-regression", json=points_request(x=[0.0, 1.0], y=[0, 1], threshold=0.500001)).json()

    assert at_boundary["predictions"]["probabilities"] == above_boundary["predictions"]["probabilities"] == [0.5, 0.5]
    assert at_boundary["predictions"]["predicted_classes"] == [1, 1]
    assert above_boundary["predictions"]["predicted_classes"] == [0, 0]
    assert at_boundary["model"] == above_boundary["model"]
    assert at_boundary["training"] == above_boundary["training"]


def test_threshold_does_not_change_fitted_values_after_training() -> None:
    body = points_request(x=[-2.0, -1.0, 1.0, 2.0], y=[0, 0, 1, 1], steps=10, learning_rate=0.2)
    lower_threshold = {**body, "threshold": 0.3}
    higher_threshold = {**body, "threshold": 0.8}
    lower = client.post("/api/v1/experiments/logistic-regression", json=lower_threshold).json()
    higher = client.post("/api/v1/experiments/logistic-regression", json=higher_threshold).json()

    assert lower["predictions"]["probabilities"] == higher["predictions"]["probabilities"]
    assert lower["model"] == higher["model"]
    assert lower["training"] == higher["training"]
    assert lower["predictions"]["predicted_classes"] != higher["predictions"]["predicted_classes"]


def test_confusion_matrix_and_classification_metrics_are_consistent() -> None:
    response = client.post("/api/v1/experiments/logistic-regression", json=points_request()).json()

    assert response["predictions"]["predicted_classes"] == [1, 1, 1, 1]
    assert response["metrics"]["confusion_matrix"] == {
        "true_positive": 2,
        "true_negative": 0,
        "false_positive": 2,
        "false_negative": 0,
    }
    assert response["metrics"]["accuracy"] == pytest.approx(0.5)
    assert response["metrics"]["precision"] == pytest.approx(0.5)
    assert response["metrics"]["recall"] == pytest.approx(1.0)
    assert response["metrics"]["f1"] == pytest.approx(2 / 3)


def test_confusion_matrix_tracks_all_four_outcomes() -> None:
    body = points_request(x=[-2.0, -1.0, 1.0, 2.0], y=[0, 1, 0, 1])
    body["parameters"].update(initial_coefficient=1.0)
    response = client.post("/api/v1/experiments/logistic-regression", json=body).json()

    assert response["predictions"]["predicted_classes"] == [0, 0, 1, 1]
    assert response["metrics"]["confusion_matrix"] == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
    }
    assert response["metrics"]["accuracy"] == pytest.approx(0.5)
    assert response["metrics"]["precision"] == pytest.approx(0.5)
    assert response["metrics"]["recall"] == pytest.approx(0.5)
    assert response["metrics"]["f1"] == pytest.approx(0.5)


def test_zero_division_metrics_are_zero_and_single_class_inline_data_is_supported() -> None:
    response = client.post(
        "/api/v1/experiments/logistic-regression",
        json=points_request(x=[-1.0, 1.0], y=[0, 0], threshold=0.500001),
    ).json()

    assert response["dataset"]["y"] == [0, 0]
    assert response["metrics"]["confusion_matrix"] == {
        "true_positive": 0,
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
    }
    assert response["metrics"]["accuracy"] == 1.0
    assert response["metrics"]["precision"] == 0.0
    assert response["metrics"]["recall"] == 0.0
    assert response["metrics"]["f1"] == 0.0


def test_default_synthetic_request_reproduces_both_classes() -> None:
    request = default_synthetic_request()
    first = client.post("/api/v1/experiments/logistic-regression", json=request)
    second = client.post("/api/v1/experiments/logistic-regression", json=request)

    assert first.status_code == 200
    assert first.json() == second.json()
    effective_dataset = first.json()["dataset"]
    assert set(effective_dataset["y"]) == {0, 1}
    assert all(-4.0 <= value < 4.0 for value in effective_dataset["x"])


def test_generated_single_class_dataset_returns_structured_actionable_error() -> None:
    request = default_synthetic_request()
    request["dataset"].update(true_coefficient=0.0, true_intercept=-1_000_000.0, n_samples=10)
    response = client.post("/api/v1/experiments/logistic-regression", json=request)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "SINGLE_CLASS_DATASET"
    assert response.json()["error"]["details"] == {"class_label": 0, "n_samples": 10}
    assert "change the seed or generator settings" in response.json()["error"]["message"]


def test_invalid_dimensions_labels_numeric_ranges_and_training_values_are_rejected() -> None:
    invalid_mutations = [
        lambda body: body["dataset"].update(y=[0]),
        lambda body: body["dataset"].update(y=[0, 0.5, 1, 1]),
        lambda body: body["dataset"].update(y=[0, 1, 2, 1]),
        lambda body: body["dataset"].update(x=[0.0, 1_000_001.0]),
        lambda body: body["parameters"].update(initial_coefficient=float("nan")),
        lambda body: body["parameters"].update(learning_rate=0.0),
        lambda body: body["parameters"].update(learning_rate=1.01),
        lambda body: body["parameters"].update(steps=501),
        lambda body: body.update(threshold=-0.01),
        lambda body: body.update(threshold=1.01),
        lambda body: body.update(seed=4_294_967_296),
        lambda body: body["dataset"].update(x=[0.0] * 1001, y=[0] * 1001),
    ]
    for mutate in invalid_mutations:
        body = points_request()
        mutate(body)
        response = client.post(
            "/api/v1/experiments/logistic-regression",
            content=json.dumps(body, allow_nan=True),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_invalid_synthetic_range_and_sample_bounds_are_rejected() -> None:
    request = default_synthetic_request()
    request["dataset"].update(x_min=2.0, x_max=2.0)
    response = client.post("/api/v1/experiments/logistic-regression", json=request)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    request = default_synthetic_request()
    request["dataset"].update(n_samples=1)
    response = client.post("/api/v1/experiments/logistic-regression", json=request)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_malformed_json_returns_structured_validation_error() -> None:
    response = client.post(
        "/api/v1/experiments/logistic-regression",
        content=b'{"dataset":',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["details"]["fields"]


def test_exactly_one_mib_request_is_accepted() -> None:
    body = json.dumps(points_request()).encode("utf-8")
    body += b" " * (1024 * 1024 - len(body))
    assert len(body) == 1024 * 1024

    response = client.post(
        "/api/v1/experiments/logistic-regression",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200


def test_oversized_request_returns_structured_413() -> None:
    response = client.post(
        "/api/v1/experiments/logistic-regression",
        content=b" " * (1024 * 1024 + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_engine_reports_non_finite_intermediate_as_divergence() -> None:
    with pytest.raises(LogisticNumericalDivergenceError) as error:
        train_logistic_regression(
            [1e308], [1.0], initial_coefficient=1e308,
            initial_intercept=0.0, learning_rate=0.1, steps=1,
        )

    assert (error.value.step, error.value.value) == (0, "logits")


def test_finite_gradient_terms_do_not_false_diverge_during_aggregation() -> None:
    with pytest.raises(LogisticNumericalDivergenceError) as error:
        train_logistic_regression(
            [1.7e308] * 4, [0.0] * 4, initial_coefficient=0.0,
            initial_intercept=0.0, learning_rate=0.1, steps=1,
        )

    # The true mean gradient is finite; divergence occurs later when the
    # updated coefficient produces an actually non-finite logit.
    assert (error.value.step, error.value.value) == (1, "logits")


def test_api_maps_engine_divergence_to_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def diverge(_request):
        raise LogisticNumericalDivergenceError(3, "gradient_coefficient")

    monkeypatch.setattr(logistic_route, "run_logistic_regression_experiment", diverge)
    response = client.post("/api/v1/experiments/logistic-regression", json=points_request())

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "NUMERICAL_DIVERGENCE",
        "message": "Logistic Regression produced a non-finite intermediate value.",
        "details": {"step": 3, "value": "gradient_coefficient"},
    }


def test_endpoint_success_response_is_finite_and_has_complete_history() -> None:
    response = client.post("/api/v1/experiments/logistic-regression", json=points_request(steps=3))

    assert response.status_code == 200
    payload = response.json()
    json.dumps(payload, allow_nan=False)
    assert payload["algorithm"] == "logistic_regression"
    assert len(payload["predictions"]["probabilities"]) == len(payload["dataset"]["x"])
    assert len(payload["predictions"]["predicted_classes"]) == len(payload["dataset"]["y"])
    assert payload["training"]["steps_completed"] == 3
    assert len(payload["training"]["loss_history"]) == 4
    assert payload["metrics"]["binary_cross_entropy"] == pytest.approx(payload["training"]["loss_history"][-1])


def test_threshold_and_seed_are_echoed_for_inline_data() -> None:
    response = client.post(
        "/api/v1/experiments/logistic-regression",
        json=points_request(x=[-1.0, 1.0], y=[0, 1], threshold=0.75),
    )

    assert response.status_code == 200
    assert response.json()["threshold"] == 0.75
    assert response.json()["seed"] == 42


@pytest.mark.parametrize("label", [True, False, "0", "1", math.nan, math.inf, -math.inf])
def test_invalid_binary_label_has_structured_error(label):
    body = points_request(x=[0], y=[label])
    response = client.post("/api/v1/experiments/logistic-regression",
                           content=json.dumps(body), headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_binary_label_validates_integer_membership_without_narrowing():
    import sys

    outside_float_range = int(sys.float_info.max) * 2
    with pytest.raises(ValidationError, match="labels must be exactly 0 or 1"):
        LogisticRegressionRequest.model_validate(points_request(x=[0], y=[outside_float_range]))


def test_oversized_integer_label_returns_field_validation_error() -> None:
    body = json.dumps(points_request(x=[0], y=[10**399]), allow_nan=False).encode("ascii")
    assert len(body) < 1024
    response = client.post(
        "/api/v1/experiments/logistic-regression",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["message"] == "Request validation failed."
    assert any(
        field["field"].endswith("y.0") and "exactly 0 or 1" in field["reason"]
        for field in error["details"]["fields"]
    )


@pytest.mark.parametrize("label", [0, 1, 0.0, 1.0])
def test_binary_numeric_labels_are_accepted(label):
    response = client.post("/api/v1/experiments/logistic-regression", json=points_request(x=[0], y=[label]))
    assert response.status_code == 200
    assert response.json()["dataset"]["y"] == [int(label)]
