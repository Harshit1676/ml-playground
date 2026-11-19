"""Typed request and response contracts for binary Logistic Regression."""

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

BoundedNumber = Annotated[float, Field(allow_inf_nan=False, ge=-1_000_000, le=1_000_000)]
DatasetNumber = BoundedNumber
Threshold = Annotated[float, Field(allow_inf_nan=False, ge=0, le=1)]
Probability = Annotated[float, Field(allow_inf_nan=False, ge=0, le=1)]
FiniteNonnegative = Annotated[float, Field(allow_inf_nan=False, ge=0)]
MetricValue = Annotated[float, Field(allow_inf_nan=False, ge=0, le=1)]


def _parse_binary_label(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("labels must be numeric 0 or 1")
    if not math.isfinite(float(value)) or value not in (0, 1):
        raise ValueError("labels must be exactly 0 or 1")
    return int(value)


BinaryLabel = Annotated[int, BeforeValidator(_parse_binary_label)]


class PointsDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["points"]
    x: list[DatasetNumber] = Field(min_length=1, max_length=1000)
    y: list[BinaryLabel] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def matching_dimensions(self) -> "PointsDataset":
        if len(self.x) != len(self.y):
            raise ValueError("x and y must contain the same number of observations")
        return self


class SyntheticLogisticDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["synthetic_logistic"]
    n_samples: int = Field(strict=True, ge=2, le=1000)
    true_coefficient: BoundedNumber
    true_intercept: BoundedNumber
    x_min: BoundedNumber
    x_max: BoundedNumber

    @model_validator(mode="after")
    def valid_range(self) -> "SyntheticLogisticDataset":
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be less than x_max")
        return self


Dataset = Annotated[PointsDataset | SyntheticLogisticDataset, Field(discriminator="kind")]


class LogisticRegressionParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_coefficient: BoundedNumber
    initial_intercept: BoundedNumber
    learning_rate: Annotated[float, Field(allow_inf_nan=False, gt=0, le=1)]
    steps: int = Field(strict=True, ge=0, le=500)


class LogisticRegressionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Dataset
    parameters: LogisticRegressionParameters
    threshold: Threshold = 0.5
    seed: int = Field(strict=True, ge=0, le=4_294_967_295)


class EffectiveDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: list[DatasetNumber]
    y: list[BinaryLabel]


class ModelParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coefficient: float = Field(allow_inf_nan=False)
    intercept: float = Field(allow_inf_nan=False)


class Predictions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probabilities: list[Probability]
    predicted_classes: list[BinaryLabel]


class ConfusionMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    binary_cross_entropy: FiniteNonnegative
    confusion_matrix: ConfusionMatrix
    accuracy: MetricValue
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue


class TrainingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps_completed: int = Field(ge=0, le=500)
    loss_history: list[FiniteNonnegative]


class LogisticRegressionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["logistic_regression"] = "logistic_regression"
    seed: int
    dataset: EffectiveDataset
    parameters: LogisticRegressionParameters
    threshold: Threshold
    model: ModelParameters
    predictions: Predictions
    metrics: Metrics
    training: TrainingSummary

    @model_validator(mode="after")
    def consistent_result_sizes_and_labels(self) -> "LogisticRegressionResponse":
        count = len(self.dataset.x)
        if count == 0 or len(self.dataset.y) != count:
            raise ValueError("effective dataset arrays must have the same non-zero length")
        if len(self.predictions.probabilities) != count or len(self.predictions.predicted_classes) != count:
            raise ValueError("prediction arrays must match the observation count")
        if len(self.training.loss_history) != self.training.steps_completed + 1:
            raise ValueError("loss_history must contain the initial loss and each completed update")

        expected_classes = [int(probability >= self.threshold) for probability in self.predictions.probabilities]
        if self.predictions.predicted_classes != expected_classes:
            raise ValueError("predicted classes must use the returned threshold")

        matrix = self.metrics.confusion_matrix
        if matrix.true_positive + matrix.true_negative + matrix.false_positive + matrix.false_negative != count:
            raise ValueError("confusion matrix counts must sum to the observation count")
        return self
