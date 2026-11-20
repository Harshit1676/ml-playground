from app.ml.clustered_dataset import generate_clustered_points
from app.ml.kmeans import fit_kmeans
from app.schemas.kmeans import (
    ClusterModel, GeneratedPoints, IterationSnapshot, KMeansRequest, KMeansResponse, Metrics, TrainingSummary,
)


def run_kmeans_experiment(request: KMeansRequest) -> KMeansResponse:
    dataset = request.dataset
    if dataset.kind == "synthetic_clusters":
        points = generate_clustered_points(dataset.centers, dataset.n_samples, dataset.spread, dataset.generation_seed)
        dataset = GeneratedPoints(**dataset.model_dump(), points=points.tolist())
    result = fit_kmeans(
        dataset.points,
        k=request.parameters.k,
        max_iterations=request.parameters.max_iterations,
        seed=request.seed,
    )
    return KMeansResponse(
        dataset=dataset,
        parameters=request.parameters,
        seed=request.seed,
        model=ClusterModel(centroids=result.centroids.tolist()),
        assignments=result.assignments.tolist(),
        metrics=Metrics(inertia=result.snapshots[-1].inertia),
        training=TrainingSummary(
            initial_centroids=result.initial_centroids.tolist(),
            iterations_completed=result.snapshots[-1].iteration,
            converged=result.converged,
            inertia_history=list(result.inertia_history),
            snapshots=[
                IterationSnapshot(
                    iteration=state.iteration,
                    centroids=state.centroids.tolist(),
                    assignments=state.assignments.tolist(),
                    inertia=state.inertia,
                )
                for state in result.snapshots
            ],
        ),
    )
