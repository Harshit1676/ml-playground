from math import dist, fsum

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.cluster import KMeans

from app.main import create_app
from app.ml.kmeans import fit_kmeans

client = TestClient(create_app())
endpoint = "/api/v1/experiments/k-means"


def experiment(points, k=2, iterations=10, seed=42):
    return {
        "dataset": {"kind": "points", "points": points},
        "parameters": {"k": k, "max_iterations": iterations},
        "seed": seed,
    }


def test_api_hand_calculated_2d_fixture_and_seed_repeatability():
    # Rows 0 and 3 are sampled. Pairs have means (1,2) and (9,10).
    request = experiment([[0, 0], [2, 4], [8, 8], [10, 12]], seed=42)
    response = client.post(endpoint, json=request)
    assert response.status_code == 200
    result = response.json()
    assert result == client.post(endpoint, json=request).json()
    assert result == {
        **request,
        "algorithm": "kmeans",
        "model": {"centroids": [[1, 2], [9, 10]]},
        "assignments": [0, 0, 1, 1],
        "metrics": {"inertia": 20},
        "training": {
            "initial_centroids": [[0, 0], [10, 12]],
            "iterations_completed": 1,
            "converged": True,
            "inertia_history": [40, 20],
            "snapshots": [
                {"iteration": 0, "centroids": [[0, 0], [10, 12]], "assignments": [0, 0, 1, 1], "inertia": 40},
                {"iteration": 1, "centroids": [[1, 2], [9, 10]], "assignments": [0, 0, 1, 1], "inertia": 20},
            ],
        },
    }
    other = client.post(endpoint, json={**request, "seed": 1}).json()
    assert other["training"]["initial_centroids"] != result["training"]["initial_centroids"]


def test_iteration_cap_returns_assignments_and_inertia_for_the_returned_centers():
    points = [[0, 0], [1, 0], [3, 0], [9, 0], [10, 0]]
    result = fit_kmeans(points, k=2, max_iterations=1, seed=0)
    assert not result.converged
    np.testing.assert_array_equal(result.centroids, [[3.25, 0], [10, 0]])
    np.testing.assert_array_equal(result.assignments, [0, 0, 0, 1, 1])
    assert result.inertia_history == (181, 16.6875)
    complete = fit_kmeans(points, k=2, max_iterations=10, seed=0)
    assert complete.converged
    assert len(complete.inertia_history) == 3
    assert np.all(np.diff(complete.inertia_history) <= 0)


def test_equidistant_point_uses_lower_cluster_index():
    result = fit_kmeans([[-2, 0], [0, 0], [2, 0], [2, 0]], k=2, max_iterations=1, seed=42)
    np.testing.assert_array_equal(result.centroids, [[-1, 0], [2, 0]])
    assert result.assignments[1] == 0


def test_empty_clusters_keep_centers_and_ids():
    result = fit_kmeans([[3, -1]] * 3, k=3, max_iterations=5, seed=7)
    np.testing.assert_array_equal(result.centroids, [[3, -1]] * 3)
    np.testing.assert_array_equal(result.assignments, [0, 0, 0])
    assert result.converged and result.inertia_history == (0, 0)


def test_tiny_distinct_points_do_not_become_a_false_distance_tie():
    points = [[0, 0], [1e-200, 1e-200]]
    result = fit_kmeans(points, k=2, max_iterations=1, seed=42)
    np.testing.assert_array_equal(result.centroids[result.assignments], points)


@pytest.mark.parametrize("limit", [1, 3, 10])
def test_replay_snapshots_with_duplicate_coordinates_and_empty_centers(limit):
    # Five distinct coordinates for four clusters, but seed 2 selects three zero rows.
    points = [[x, 0] for x in [0, 0, 0, 1, 3, 9, 10]]
    request = experiment(points, k=4, iterations=limit, seed=2)
    result = client.post(endpoint, json=request).json()
    assert client.post(endpoint, json=request).json() == result
    training = result["training"]
    states = training["snapshots"]
    assert states[0]["centroids"] == [[0, 0], [1, 0], [0, 0], [0, 0]]
    assert training["iterations_completed"] == min(limit, 3) == len(states) - 1
    assert training["converged"] is (limit >= 3)
    assert states[-1]["centroids"] == result["model"]["centroids"]
    assert states[-1]["assignments"] == result["assignments"]
    assert states[-1]["inertia"] == result["metrics"]["inertia"]
    counts_by_step = [[3, 4, 0, 0], [4, 3, 0, 0], [2, 2, 3, 0], [2, 2, 3, 0]]
    for iteration, state in enumerate(states):
        centers = state["centroids"]
        nearest = [min(range(4), key=lambda i: dist(point, centers[i])) for point in points]
        inertia = fsum(dist(point, centers[i]) ** 2 for point, i in zip(points, nearest))
        assert state["iteration"] == iteration
        assert state["assignments"] == nearest
        assert [nearest.count(i) for i in range(4)] == counts_by_step[iteration]
        assert state["inertia"] == pytest.approx(inertia)
        assert training["inertia_history"][iteration] == state["inertia"]
        if iteration:
            previous = states[iteration - 1]
            for i, center in enumerate(centers):
                members = [p for p, label in zip(points, previous["assignments"]) if label == i]
                expected = ([fsum(axis) / len(members) for axis in zip(*members)]
                            if members else previous["centroids"][i])
                assert center == pytest.approx(expected)
    if limit == 1:
        # Last reassignment moves x=1 into cluster 0. Its new mean would be 0.25.
        assert states[-1]["centroids"][0] == [0, 0]


def test_identical_decimal_points_stay_exactly_stationary():
    request = experiment([[0.1, -0.1]] * 3, k=3)
    result = client.post(endpoint, json=request).json()
    assert result["training"]["converged"]
    states = result["training"]["snapshots"]
    assert len(states) == 2
    for state in states:
        assert state["centroids"] == request["dataset"]["points"]
        assert state["assignments"] == [0, 0, 0]
        assert state["inertia"] == 0


def test_maximum_length_replay_fits_in_one_mib():
    t = np.arange(1000) / 999
    points = np.column_stack((t * 999999.123456789, t * -38231.2345678901))
    response = client.post(endpoint, json=experiment(points.tolist(), k=16, iterations=100, seed=3))
    assert response.status_code == 200
    training = response.json()["training"]
    assert not training["converged"]
    assert len(training["snapshots"]) == 101
    assert len(response.content) < 2**20


def test_squared_errors_can_add_to_representable_inertia_even_when_individually_tiny():
    points = [[1e-162, 0], [-1e-162, 0]] * 50
    result = fit_kmeans(points, k=1, max_iterations=1, seed=0)
    assert result.inertia_history[-1] == float("1e-322")


def test_matches_sklearn_lloyd_from_same_centroids():
    points = np.array([[0, 0], [1, 2], [3, 1], [9, 9], [10, 12]], dtype=np.float64)
    result = fit_kmeans(points, k=2, max_iterations=20, seed=0)
    reference = KMeans(n_clusters=2, init=result.initial_centroids, n_init=1,
                       algorithm="lloyd", tol=0, max_iter=20).fit(points)
    assert result.converged
    np.testing.assert_allclose(result.centroids, reference.cluster_centers_, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(result.assignments, reference.labels_)
    assert result.inertia_history[-1] == pytest.approx(reference.inertia_, abs=1e-12, rel=0)


@pytest.mark.parametrize("request_body", [
    experiment([]), experiment([[0]], k=1), experiment([[0, 1, 2]], k=1),
    experiment([[True, 0]], k=1), experiment([["1", 0]], k=1),
    experiment([[0, 1_000_001]], k=1), experiment([[0, 0]] * 1001),
    experiment([[0, 0]]), experiment([[0, 0]], k=0), experiment([[0, 0]] * 17, k=17),
    experiment([[0, 0]], k=1, iterations=0), experiment([[0, 0]], k=1, iterations=101),
    experiment([[0, 0]], k=1, seed=-1), experiment([[0, 0]], k=1, seed=2**32),
    {**experiment([[0, 0]], k=1), "dataset": {"kind": "points", "points": [[0, 0]], "labels": [0]}},
])
def test_invalid_requests_use_structured_validation_errors(request_body):
    response = client.post(endpoint, json=request_body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_limits_accept_extreme_coordinates_and_reject_oversize_body():
    points = [[-1_000_000, 1_000_000], [1_000_000, -1_000_000]] * 500
    response = client.post(endpoint, json=experiment(points, k=1, iterations=100, seed=2**32 - 1))
    assert response.status_code == 200
    assert response.json()["metrics"]["inertia"] == 2e15
    assert client.post(endpoint, content=b" " * (2**20 + 1)).status_code == 413


def test_nonfinite_coordinate_returns_serializable_validation_error():
    response = client.post(
        endpoint,
        content='{"dataset":{"kind":"points","points":[[1e999,0]]},'
                '"parameters":{"k":1,"max_iterations":1},"seed":0}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def synthetic_request(**changes):
    return {
        "dataset": {
            "kind": "synthetic_clusters", "centers": [[-4, 2], [6, -3]], "spread": 0.25,
            "generation_seed": 17, "n_samples": 30, **changes,
        },
        "parameters": {"k": 3, "max_iterations": 10},
        "seed": 42,
    }


def test_seeds_control_separate_stages_and_generated_points_can_be_replayed_inline():
    request = synthetic_request()
    response = client.post(endpoint, json=request)
    assert response.status_code == 200
    run = response.json()
    assert client.post(endpoint, json=request).json() == run
    points = run["dataset"]["points"]
    assert run["dataset"] == {**request["dataset"], "points": points}
    assert len(points) == 30
    assert all(any(max(abs(x - cx), abs(y - cy)) <= 0.25 for cx, cy in request["dataset"]["centers"])
               for x, y in points)
    other_init = client.post(endpoint, json={**request, "seed": 7}).json()
    assert other_init["dataset"] == run["dataset"]
    assert other_init["training"]["initial_centroids"] != run["training"]["initial_centroids"]

    other_data = client.post(endpoint, json=synthetic_request(generation_seed=18)).json()
    assert other_data["dataset"]["points"] != points
    inline = {**request, "dataset": {"kind": "points", "points": points}}
    assert client.post(endpoint, json=inline).json() == {**run, "dataset": inline["dataset"]}


@pytest.mark.parametrize(("count", "centers", "spread", "generation_seed"), [
    (1, [[1e6, -1e6]], 0, 0),
    (1000, [[0, 0]] * 16, 1e6, 2**32 - 1),
])
def test_synthetic_limits(count, centers, spread, generation_seed):
    request = synthetic_request(n_samples=count, centers=centers, spread=spread, generation_seed=generation_seed)
    request["parameters"]["k"] = 1
    response = client.post(endpoint, json=request)
    assert response.status_code == 200
    points = response.json()["dataset"]["points"]
    assert len(points) == count
    assert np.abs(points).max() <= 1e6
    if spread == 0:
        assert points == centers


@pytest.mark.parametrize("changes", [
    {"n_samples": 0}, {"n_samples": 1001}, {"n_samples": 3.5}, {"n_samples": 2},  # k = 3
    {"centers": []}, {"centers": [[0, 0]] * 17}, {"centers": [[1]]}, {"centers": [[1, 2, 3]]},
    {"centers": [[1_000_001, 0]]}, {"centers": [[0, -1e6]]},  # spread would exceed the bound
    {"spread": -1}, {"spread": 1_000_001}, {"spread": True},
    {"generation_seed": -1}, {"generation_seed": 2**32}, {"generation_seed": True},
    {"labels": [1]}, {"points": [[0, 0]]},
])
def test_synthetic_invalid_requests(changes):
    response = client.post(endpoint, json=synthetic_request(**changes))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_canonical_route_and_discriminated_openapi_contract():
    spec = client.get("/openapi.json").json()
    assert "/api/v1/experiments/kmeans" not in spec["paths"]
    assert client.post("/api/v1/experiments/kmeans", json=synthetic_request()).status_code == 404
    operation = spec["paths"][endpoint]["post"]
    contracts = [operation["requestBody"], operation["responses"]["200"]]
    for contract in contracts:
        ref = contract["content"]["application/json"]["schema"]["$ref"]
        dataset = spec["components"]["schemas"][ref.rsplit("/", 1)[1]]["properties"]["dataset"]
        assert dataset["discriminator"]["propertyName"] == "kind"
        assert set(dataset["discriminator"]["mapping"]) == {"points", "synthetic_clusters"}
