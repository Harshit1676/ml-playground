from app.ml.kmeans import fit_kmeans
from app.schemas.kmeans import ClusterModel, IterationSnapshot, KMeansRequest, KMeansResponse, Metrics, TrainingSummary


def run_kmeans_experiment(request: KMeansRequest) -> KMeansResponse:
    result = fit_kmeans(
        request.dataset.points,
        k=request.parameters.k,
        max_iterations=request.parameters.max_iterations,
        seed=request.seed,
    )
    return KMeansResponse(
        dataset=request.dataset,
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
