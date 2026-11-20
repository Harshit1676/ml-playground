"""REST contract for unlabeled two-dimensional K-Means experiments."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Coordinate = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1_000_000, le=1_000_000)]
Point = tuple[Coordinate, Coordinate]
Inertia = Annotated[float, Field(allow_inf_nan=False, ge=0)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PointsDataset(ContractModel):
    kind: Literal["points"]
    points: list[Point] = Field(min_length=1, max_length=1000)


class KMeansParameters(ContractModel):
    k: int = Field(strict=True, ge=1, le=16)
    max_iterations: int = Field(strict=True, ge=1, le=100)


class KMeansRequest(ContractModel):
    dataset: PointsDataset
    parameters: KMeansParameters
    seed: int = Field(strict=True, ge=0, le=4_294_967_295)

    @model_validator(mode="after")
    def enough_observations(self) -> "KMeansRequest":
        if self.parameters.k > len(self.dataset.points):
            raise ValueError("k must not exceed the number of observations")
        return self


class ClusterModel(ContractModel):
    centroids: list[Point]


class Metrics(ContractModel):
    inertia: Inertia


class IterationSnapshot(ContractModel):
    iteration: int = Field(ge=0, le=100)
    centroids: list[Point] = Field(min_length=1, max_length=16)
    assignments: list[int] = Field(min_length=1, max_length=1000)
    inertia: Inertia


class TrainingSummary(ContractModel):
    initial_centroids: list[Point]
    iterations_completed: int = Field(ge=1, le=100)
    converged: bool
    inertia_history: list[Inertia]
    snapshots: list[IterationSnapshot] = Field(min_length=2, max_length=101)


class KMeansResponse(KMeansRequest):
    algorithm: Literal["kmeans"] = "kmeans"
    model: ClusterModel
    assignments: list[int]
    metrics: Metrics
    training: TrainingSummary

    @model_validator(mode="after")
    def consistent_result(self) -> "KMeansResponse":
        k = self.parameters.k
        training = self.training
        if len(self.model.centroids) != k or len(training.initial_centroids) != k:
            raise ValueError("expected k initial and final centroids")
        if len(self.assignments) != len(self.dataset.points) or any(i < 0 or i >= k for i in self.assignments):
            raise ValueError("expected one valid cluster ID per observation")
        if training.iterations_completed > self.parameters.max_iterations:
            raise ValueError("iteration limit exceeded")
        if len(training.inertia_history) != training.iterations_completed + 1:
            raise ValueError("expected initial inertia and one entry per update")
        if self.metrics.inertia != training.inertia_history[-1]:
            raise ValueError("final inertia must match the history")
        states = training.snapshots
        if len(states) != len(training.inertia_history):
            raise ValueError("expected one snapshot per inertia entry")
        for iteration, state in enumerate(states):
            if state.iteration != iteration or state.inertia != training.inertia_history[iteration]:
                raise ValueError("snapshot iteration and inertia must match the history")
            if len(state.centroids) != k or len(state.assignments) != len(self.dataset.points):
                raise ValueError("snapshot dimensions must match the experiment")
            if any(label < 0 or label >= k for label in state.assignments):
                raise ValueError("invalid snapshot cluster ID")
        if states[0].centroids != training.initial_centroids:
            raise ValueError("initial centroids must match snapshot zero")
        if states[-1].centroids != self.model.centroids or states[-1].assignments != self.assignments:
            raise ValueError("final model and assignments must match the last snapshot")
        if training.converged != (states[-1].assignments == states[-2].assignments):
            raise ValueError("convergence must reflect stable final assignments")
        if not training.converged and training.iterations_completed != self.parameters.max_iterations:
            raise ValueError("an unconverged run must exhaust the iteration limit")
        return self
