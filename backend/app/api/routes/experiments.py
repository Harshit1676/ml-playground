from app.api.admission import experiment_router
from app.schemas.errors import ErrorResponse
from app.schemas.linear_regression import LinearRegressionRequest, LinearRegressionResponse
from app.services.linear_regression import run_linear_regression_experiment

router = experiment_router()


@router.post(
    "/experiments/linear-regression",
    response_model=LinearRegressionResponse,
    responses={
        413: {"model": ErrorResponse, "description": "Request body exceeds 1 MiB."},
        422: {"model": ErrorResponse, "description": "Invalid input or numerical divergence."},
    },
    tags=["experiments"],
)
def post_linear_regression(request: LinearRegressionRequest) -> LinearRegressionResponse:
    return run_linear_regression_experiment(request)
