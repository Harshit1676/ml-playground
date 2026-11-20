"""Lloyd's algorithm for bounded two-dimensional observations."""

from dataclasses import dataclass
from math import fsum

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IterationState:
    iteration: int
    centroids: FloatArray
    assignments: NDArray[np.intp]
    inertia: float


@dataclass(frozen=True)
class KMeansResult:
    snapshots: tuple[IterationState, ...]
    converged: bool

    @property
    def initial_centroids(self) -> FloatArray:
        return self.snapshots[0].centroids

    @property
    def centroids(self) -> FloatArray:
        return self.snapshots[-1].centroids

    @property
    def assignments(self) -> NDArray[np.intp]:
        return self.snapshots[-1].assignments

    @property
    def inertia_history(self) -> tuple[float, ...]:
        return tuple(state.inertia for state in self.snapshots)


def _assign(points: FloatArray, centroids: FloatArray) -> tuple[NDArray[np.intp], float]:
    offsets = points[:, None, :] - centroids
    squared = np.sum(offsets * offsets, axis=2)
    assignments = squared.argmin(axis=1)

    # Squaring a tiny distance can erase the distinction between two centers.
    # Compare the lengths on these rows, preserving argmin's first-index tie rule.
    tiny_rows = np.sum(squared < np.finfo(np.float64).tiny, axis=1) > 1
    if np.any(tiny_rows):
        small = np.ldexp(offsets[tiny_rows], 512)
        assignments[tiny_rows] = np.hypot(small[:, :, 0], small[:, :, 1]).argmin(axis=1)

    residuals = points - centroids[assignments]
    # Accumulate in power-of-two scaled units before rounding the final SSE.
    scale = np.abs(residuals).max()
    _, exponent = np.frexp(scale)
    normalized = np.ldexp(residuals, -exponent)
    inertia = np.ldexp(np.sum(normalized * normalized), 2 * exponent)
    return assignments, float(inertia)


def fit_kmeans(points: ArrayLike, *, k: int, max_iterations: int, seed: int) -> KMeansResult:
    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) == 0:
        raise ValueError("provide a nonempty matrix of two-dimensional points")
    if not np.isfinite(data).all() or np.any(np.abs(data) > 1_000_000):
        raise ValueError("coordinates must be finite and within +/-1,000,000")
    if not 1 <= k <= len(data) or max_iterations < 1:
        raise ValueError("require 1 <= k <= observation count and at least one iteration")

    indices = np.random.default_rng(seed).choice(len(data), size=k, replace=False)
    centroids = data[indices]
    assignments, inertia = _assign(data, centroids)
    history = [IterationState(0, centroids, assignments, inertia)]
    for iteration in range(1, max_iterations + 1):
        centroids = centroids.copy()
        for cluster in range(k):
            members = data[assignments == cluster]
            if len(members):
                # Accurate sums also handle cancellation between opposite coordinates.
                centroids[cluster] = [
                    column[0] if np.all(column == column[0]) else fsum(column) / len(members)
                    for column in members.T
                ]
        updated, inertia = _assign(data, centroids)
        history.append(IterationState(iteration, centroids, updated, inertia))
        converged = np.array_equal(updated, assignments)
        assignments = updated
        if converged:
            break

    return KMeansResult(tuple(history), converged)
