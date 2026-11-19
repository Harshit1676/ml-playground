# Architecture

## 1. Goals and technology

- **Frontend:** React, Vite, TypeScript, Tailwind CSS, Plotly.js
- **Backend:** Python 3.11+, FastAPI, Pydantic, NumPy, scikit-learn
- **Testing:** Vitest, React Testing Library, pytest, API contract/integration tests
- **Deployment:** static frontend hosting and a separate FastAPI service
- **MVP constraints:** stateless backend; no authentication, database, Redis, Celery, or AI chatbot/LLM

Frontend owns interaction, controls, visualization, and explanation presentation. All dataset generation, ML training, prediction, metrics, and learning-step calculations happen in a separate Python engine called by FastAPI. React must not implement model computation.

## 2. Monorepo layout

Planned structure; application code is not yet scaffolded.

```text
frontend/
  src/
    components/         Controls and shared presentation
    features/           Algorithm experiences and UI state
    lib/api/             Typed API client and transport errors
    lib/visualization/  Plotly adapters and chart-data transforms
    lib/explanations/   Learner-facing explanation presentation
    types/               Frontend domain and API types
  tests/                Vitest + React Testing Library

backend/
  app/
    main.py             FastAPI app and middleware
    api/v1/routes/      Thin HTTP endpoint handlers
    schemas/            Pydantic request, response, and error schemas
    services/           Experiment orchestration and response assembly
    ml/                 Dataset generation, algorithms, metrics, steps
  tests/                pytest unit, route, service, and contract tests

docs/
```

Keep algorithm functions importable without starting FastAPI. Keep Plotly details out of API transport and model code.

## 3. Module responsibilities

### Frontend

- UI state owns selected algorithm, inputs, controls, seed, and pending/error state. Use explicit state transitions; do not duplicate the backend's algorithm calculations.
- API client serializes typed requests, calls configured FastAPI base URL, validates/maps responses and structured errors.
- Visualization transforms backend result data into Plotly traces; charts do not train or predict.
- Explanation presentation combines learner-facing static algorithm concepts with the returned inputs/results; it does not independently compute model output.
- Experience composition keeps chart and explanation bound to the same request. Ignore stale responses superseded by a later interaction.

### Backend layers

- **Routes:** versioned REST endpoints; bind schemas, call services, and map domain errors to HTTP. Keep HTTP handlers thin.
- **Schemas:** Pydantic request/response/error models, discriminated algorithm and dataset variants, finite-number/range/dimension/count validation, OpenAPI descriptions.
- **Services:** resolve/generate data, call the ML engine, compute/assemble metrics and response. Do not store experiment state.
- **Python ML engine:** NumPy-based data generation, model training, predictions, metrics, and explicit educational update steps. It accepts validated plain domain values and returns domain results, without FastAPI, React, or Plotly dependencies.
- **scikit-learn:** use as an appropriate reference for conventional estimator behavior and known-answer comparisons. Use transparent NumPy gradient updates when exposing learning steps; document any solver/tolerance/regularization difference.

## 4. Dataset dimensions by algorithm

- **Linear Regression V1:** one input feature `x` (scalar per observation) and one continuous target `y`. The chart plots `x` horizontally and `y` vertically. Do not describe this as a two-feature input dataset.
- **Logistic Regression:** two numeric input features and a class label per observation.
- **KNN:** two numeric input features and a class label per observation.
- **K-Means:** two numeric input features and no supplied labels; cluster assignment is a computed output.

Keep these distinctions explicit in Pydantic schema variants and frontend controls. Do not send meaningless labels for K-Means.

## 5. Linear Regression V1 REST contract

### Endpoint

`POST /api/v1/experiments/linear-regression` accepts all state for one bounded synchronous experiment. No session or server-side continuation exists. An optional generated dataset is sampled with the supplied seed; inline observations are submitted directly.

### Request

Fields: `dataset` union (`points` with parallel scalar arrays `x` and continuous `y`, or `synthetic_linear` with sample count, slope, intercept, noise standard deviation and x range); `parameters` (`initial_slope`, `initial_intercept`, `learning_rate`, `steps`); `seed` unsigned 32-bit integer.

```json
{
  "dataset": {"kind": "points", "x": [-1.0, 0.0, 1.0], "y": [-1.0, 1.0, 3.0]},
  "parameters": {"initial_slope": 0.0, "initial_intercept": 0.0, "learning_rate": 0.1, "steps": 0},
  "seed": 42
}
```

The `steps` value is the exact number of batch gradient descent updates performed from the supplied initial parameters. `steps=0` evaluates without updating. To continue one step, the client resubmits the same data and effective configuration, passes the prior response's slope/intercept as initial parameters, and sets `steps=1`. This stateless contract makes continuation explicit and reproducible.

Objective: `MSE = (1/n) Σ((slope*x_i + intercept) - y_i)^2`. At each update, calculate both gradients from the same pre-update parameters: `d_slope=(2/n)Σ((prediction_i-y_i)*x_i)` and `d_intercept=(2/n)Σ(prediction_i-y_i)`. Update simultaneously by subtracting `learning_rate * gradient`. Loss history contains initial MSE plus post-update MSE for every step, length exactly `steps + 1`.

### Response

Response returns the algorithm, seed, normalized dataset used, effective parameters, final model, predictions in observation order, named metrics, and training details.

```json
{
  "algorithm": "linear_regression",
  "seed": 42,
  "dataset": {"x": [-1.0, 0.0, 1.0], "y": [-1.0, 1.0, 3.0]},
  "parameters": {"initial_slope": 0.0, "initial_intercept": 0.0, "learning_rate": 0.1, "steps": 0},
  "model": {"slope": 0.0, "intercept": 0.0},
  "predictions": {"y_hat": [0.0, 0.0, 0.0]},
  "metrics": {"mse": 3.6666666667},
  "training": {"steps_completed": 0, "loss_history": [3.6666666667]}
}
```

### Validation

- Inline `x`/`y` arrays must have equal lengths from 1 through 1,000; exactly one feature is represented by each scalar `x`.
- All values are finite; absolute `x`, `y`, `initial_slope`, and `initial_intercept` values <= 1e6.
- `0 < learning_rate <= 1`; `0 <= steps <= 500`; `0 <= seed <= 2^32-1`.
- Synthetic data requires 2-1,000 samples, finite generator settings, `0 <= noise_std <= 1e3`, `x_min < x_max`, and absolute slope/intercept/range values <= 1e6.
- Request body <= 1 MiB. Validate in Pydantic and service before expensive work; reject instead of truncating.
- Invalid schema or values return HTTP 422 with stable error code/message and optional field details. Body overflow returns 413. Non-finite computation results return a structured computation error, never JSON NaN/Infinity.
- The bounds are initial limits. Feature scale can make a learning rate numerically unstable; detect non-finite results and decide during implementation whether to impose a scale-aware conservative bound or surface a clear error.

The seed controls synthetic feature/noise generation through a local NumPy random generator and is echoed in the response. Identical generated request plus seed yields identical data and equivalent results within a documented floating-point tolerance. Inline data is inherently fixed; the seed remains in the uniform request/response contract.

## 6. Other API shape and stateless flow

Use `/api/v1/experiments/{algorithm}` for remaining algorithm runs and explicit typed Pydantic variants. Logistic Regression/KNN schemas require two feature values and labels; K-Means requires two feature values and excludes input labels. Return normalized data, predictions/probabilities or cluster assignments/centroids, named metrics, seed/effective parameters, and bounded chart data needed to accurately render the result.

```text
Controls + current UI snapshot
        │ typed request + seed
        ▼
FastAPI route → Pydantic validation → service → Python ML engine
        ▲                                      │
        └──── response: dataset + result + metrics + step details
                         │
           frontend chart + explanation render
```

Every request carries all required data and state. The backend stores no user or experiment state and performs bounded synchronous work. The frontend binds responses to request versions and discards stale results.

## 7. Reproducibility and mathematical correctness

- Pass explicit seeds to all randomized generation/initialization; use local NumPy generators, never shared global random state.
- Document objective, optimizer, stopping behavior, initialization, distance, threshold, tie-breaking, degenerate cases, and numeric tolerance by algorithm.
- Compare against scikit-learn where meaningful, with solver/regularization/tolerance differences documented. The reference estimator does not replace transparent calculations needed for teaching.
- Define behavior for empty clusters, one-class data, constant features, singular data, ties, and invalid/non-finite values.
- Charts and explanations consume exactly the result snapshot returned for the active request.

## 8. Testing strategy

### Frontend: Vitest + React Testing Library

Test UI state, parameter bounds/serialization, API client success and errors, stale-response handling, chart-data transforms, and explanation presentation. Component tests cover control changes, accessible labels, pending/error states, and updates from response fixtures aligned with the API schemas.

### Backend: pytest

Test Pydantic constraints and error envelopes, seeded dataset generation, algorithm known-answer cases, gradients/learning-step semantics, metrics, edge cases, and repeatability. Route/service tests cover valid requests, invalid dimensions, size/work limits, status codes, and response models. Compare to scikit-learn where appropriate with explicit tolerances.

### Integration

Exercise frontend API client against a local FastAPI service for request/response schema compatibility, error mapping, seeded repeatability, boundary validation, and Linear Regression continuation via resubmitted parameters. Add one representative happy path per algorithm as each is delivered. Manual checks cover static hosting to backend CORS, keyboard/responsive behavior, and plot/result consistency.

## 9. Performance targets and benchmark policy

Linear Regression V1 goals:

- ML engine p95 <= 100 ms for 1,000 observations and 500 gradient steps.
- Full FastAPI p95 <= 250 ms for that workload, including response serialization on a documented reference backend machine.
- Response <= 1 MiB.
- Frontend p95 <= 1,000 ms from submit to updated chart on a documented local reference setup, including API round trip.

These are targets only. Before claiming an achieved target, benchmark at least 100 warmed backend requests and 30 UI interactions, record hardware, OS/runtime/library versions, workload, payload sizes, and percentile results in `docs/DEVELOPMENT_LOG.md`. No performance claim without this evidence. Do not trade away correctness or silently truncate outputs.

## 10. Deployment decisions and tradeoffs

Build the frontend as static assets with Vite for static hosting. Deploy FastAPI separately on a Python-capable service. This matches each runtime and keeps scientific computation out of the browser, with the tradeoff of two deployments, a configured API URL, CORS policy, and separate monitoring. Hosting provider, production origins, health checks, and environment variable names remain to be decided before deployment.

Stateless synchronous calls simplify MVP operations and reproducibility. Strict input/work limits are required; large datasets and long-running training are outside this design. No database or job queue is needed for the bounded MVP experiments.
