from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BoundedNumber = Annotated[float, Field(allow_inf_nan=False, ge=-1_000_000, le=1_000_000)]
DatasetNumber = Annotated[float, Field(allow_inf_nan=False, ge=-1_000_000, le=1_000_000)]


class PointsDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["points"]
    x: list[DatasetNumber] = Field(min_length=1, max_length=1000)
    y: list[DatasetNumber] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def matching_dimensions(self) -> "PointsDataset":
        if len(self.x) != len(self.y):
            raise ValueError("x and y must contain the same number of observations")
        return self


class SyntheticLinearDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["synthetic_linear"]
    n_samples: int = Field(strict=True, ge=2, le=1000)
    slope: BoundedNumber
    intercept: BoundedNumber
    noise_std: Annotated[float, Field(allow_inf_nan=False, ge=0, le=1000)]
    x_min: BoundedNumber
    x_max: BoundedNumber

    @model_validator(mode="after")
    def valid_range(self) -> "SyntheticLinearDataset":
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be less than x_max")
        return self


Dataset = Annotated[PointsDataset | SyntheticLinearDataset, Field(discriminator="kind")]


class LinearRegressionParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial_slope: BoundedNumber
    initial_intercept: BoundedNumber
    learning_rate: Annotated[float, Field(allow_inf_nan=False, gt=0, le=1)]
    steps: int = Field(strict=True, ge=0, le=500)


class LinearRegressionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Dataset
    parameters: LinearRegressionParameters
    seed: int = Field(strict=True, ge=0, le=4_294_967_295)


class EffectiveDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: list[float]
    y: list[float]


class ModelParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slope: float = Field(allow_inf_nan=False)
    intercept: float = Field(allow_inf_nan=False)


class PredictionValues(BaseModel):
    model_config = ConfigDict(extra="forbid")
    y_hat: list[float]


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mse: float = Field(allow_inf_nan=False, ge=0)


class TrainingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps_completed: int = Field(ge=0, le=500)
    loss_history: list[float]


class LinearRegressionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    algorithm: Literal["linear_regression"] = "linear_regression"
    seed: int
    dataset: EffectiveDataset
    parameters: LinearRegressionParameters
    model: ModelParameters
    predictions: PredictionValues
    metrics: Metrics
    training: TrainingSummary

    @model_validator(mode="after")
    def consistent_result_sizes(self) -> "LinearRegressionResponse":
        if len(self.dataset.x) != len(self.dataset.y) or len(self.dataset.x) != len(self.predictions.y_hat):
            raise ValueError("dataset and prediction lengths must match")
        if len(self.training.loss_history) != self.training.steps_completed + 1:
            raise ValueError("loss_history must contain the initial loss and each completed update")
        return self
