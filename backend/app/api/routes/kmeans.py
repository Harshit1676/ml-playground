from fastapi import APIRouter

from app.schemas.errors import ErrorResponse
from app.schemas.kmeans import KMeansRequest, KMeansResponse
from app.services.kmeans import run_kmeans_experiment

router = APIRouter()


@router.post(
    "/experiments/k-means",
    response_model=KMeansResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Request body exceeds 1 MiB."},
        422: {"model": ErrorResponse, "description": "Invalid K-Means experiment."},
    },
    tags=["experiments"],
)
def post_kmeans(request: KMeansRequest) -> KMeansResponse:
    return run_kmeans_experiment(request)
