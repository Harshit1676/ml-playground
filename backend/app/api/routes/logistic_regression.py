from fastapi import APIRouter

from app.schemas.errors import ErrorResponse
from app.schemas.logistic_regression import LogisticRegressionRequest, LogisticRegressionResponse
from app.services.logistic_regression import run_logistic_regression_experiment

router = APIRouter()


@router.post(
    "/experiments/logistic-regression",
    response_model=LogisticRegressionResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Request body exceeds 1 MiB."},
        422: {"model": ErrorResponse, "description": "Invalid input, single-class generated data, or numerical divergence."},
    },
    tags=["experiments"],
)
def post_logistic_regression(request: LogisticRegressionRequest) -> LogisticRegressionResponse:
    return run_logistic_regression_experiment(request)
