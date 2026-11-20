"""Generate unlabeled 2D observations for a validated cluster recipe."""

import numpy as np
from numpy.typing import NDArray


def generate_clustered_points(
    centers: list[tuple[float, float]], n_samples: int, spread: float, generation_seed: int
) -> NDArray[np.float64]:
    rng = np.random.default_rng(generation_seed)
    source_rows = rng.integers(len(centers), size=n_samples)
    offsets = rng.uniform(-spread, spread, size=(n_samples, 2))
    return np.asarray(centers, dtype=np.float64)[source_rows] + offsets
