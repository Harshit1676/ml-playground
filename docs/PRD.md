# Product Requirements: ML Playground

## 1. Summary and goals

ML Playground helps beginner-to-intermediate learners build intuition for machine learning by changing data and parameters, inspecting the result, and following clear explanations and selected learning steps. A React frontend presents the experience; a stateless FastAPI service calls a separate Python ML engine.

Goals:

- Make foundational algorithms explorable through direct manipulation.
- Show mathematically correct visual outcomes and metrics.
- Keep computation independently testable and outside React.
- Make requests reproducible and bounded without persistent user data.

## 2. Audience

Beginner-to-intermediate learners need to understand what a chart encodes, see how parameters affect a model, follow selected calculations, and recover from confusing configurations without writing code.

## 3. MVP scope

Algorithms ship in this order:

1. **Linear Regression:** V1 uses one numeric input feature `x` and one continuous target `y`. Fit a line, show predictions/residual error, and expose explicit gradient descent steps.
2. **Logistic Regression:** two numeric input features and class labels. Show probabilities and decision boundary; distinguish probabilities from thresholded labels.
3. **KNN:** two numeric input features and class labels. Show regions corresponding to selected `k` and distance rule.
4. **K-Means:** two numeric input features, no labels. Show cluster assignments and centroids; explain initialization sensitivity.

Shared capabilities include reproducible datasets, bounded parameter controls, mathematically grounded charts, plain-language explanations, typed REST interactions, useful validation errors, responsive layout, and keyboard accessibility.

Out of scope: AI chatbot/LLM, authentication, database or saved experiment history, Redis, Celery/task queues, arbitrary code execution, and algorithms beyond the four listed.

## 4. Core experience

1. Learner chooses an algorithm, starting with Linear Regression.
2. Page presents data, controls, visualization, and explanation.
3. Learner changes data or model parameters and advances a learning step where available.
4. Frontend sends a typed request; FastAPI validates and calls the Python ML engine.
5. Frontend updates the plot and explanation from the response for that exact request.
6. Reset restores defaults and seed so the baseline can be reproduced.

## 5. Functional requirements

- **FR-1 Progression:** implement Linear Regression first, then Logistic Regression, KNN, and K-Means.
- **FR-2 Dimensions:** enforce the per-algorithm feature and label dimensions defined in Section 3.
- **FR-3 Data/controls:** users can select or configure supported datasets and valid parameters; generated data is seeded.
- **FR-4 API/computation:** versioned Pydantic REST contracts call a separate Python engine for generation, training, prediction, metrics, and learning steps.
- **FR-5 Validation:** reject invalid shape, non-finite values, out-of-range parameters, oversized payloads, and excessive work with structured actionable errors.
- **FR-6 Reproducibility:** same request and seed yield equivalent results within documented floating-point tolerance.
- **FR-7 Consistent views:** chart and explanation derive from the same response; stale responses cannot replace newer interaction results.
- **FR-8 Statelessness:** each operation carries all needed data/state; server retains no experiment state.
- **FR-9 Accessibility:** controls and charts are labeled and remain usable on narrow screens and with keyboard navigation.

## 6. Linear Regression V1 API contract

### Endpoint and semantics

`POST /api/v1/experiments/linear-regression` fits one-feature ordinary linear regression using batch gradient descent on mean squared error. The endpoint accepts either bounded inline arrays or a seeded synthetic dataset. It performs exactly `steps` updates from the supplied initial slope and intercept; it does not retain state between calls. To advance one more step, the client resubmits the same dataset, returned slope/intercept as the next initial parameters, same learning rate and seed, and `steps: 1`.

For `n` observations, `prediction_i = slope * x_i + intercept` and `MSE = (1/n) Σ(prediction_i - y_i)^2`. At each step, gradients are computed from the parameters at the start of that step:

`d_slope = (2/n) Σ((prediction_i - y_i) * x_i)`

`d_intercept = (2/n) Σ(prediction_i - y_i)`

Then both parameters are updated simultaneously: `parameter_new = parameter_old - learning_rate * gradient`. `loss_history` contains initial MSE followed by MSE after each update, so its length is `steps + 1`.

### Request schema and example

Conceptual Pydantic schema: `dataset` is a discriminated union of inline points and generated linear data; `parameters` contains `initial_slope`, `initial_intercept`, `learning_rate`, and `steps`; `seed` is an unsigned 32-bit integer. Inline observations use two parallel arrays: scalar `x` (one feature) and continuous numeric `y`.

```http
POST /api/v1/experiments/linear-regression
Content-Type: application/json
```

```json
{
  "dataset": {"kind": "points", "x": [-1.0, 0.0, 1.0], "y": [-1.0, 1.0, 3.0]},
  "parameters": {"initial_slope": 0.0, "initial_intercept": 0.0, "learning_rate": 0.1, "steps": 0},
  "seed": 42
}
```

`steps: 0` evaluates the initial model without updates. For generated data, use e.g. `{"kind":"synthetic_linear","n_samples":100,"slope":2.0,"intercept":1.0,"noise_std":0.2,"x_min":-5.0,"x_max":5.0}`; the seeded NumPy generator produces the feature values and Gaussian noise.

### Response schema and example

A successful response echoes the seed and effective parameters and returns the exact dataset used, predictions for those `x` values, final fitted parameters, named metrics, and loss history. Arrays preserve observation order.

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

### Validation and errors

- Require `1 <= n <= 1000`; for inline data, `x` and `y` lengths must match and contain at least one pair.
- Every `x`, `y`, initial parameter, and learning rate must be finite. `x` and `y` values, `initial_slope`, and `initial_intercept` must have absolute value at most `1e6`.
- Require `0 < learning_rate <= 1`, `0 <= steps <= 500`, and `0 <= seed <= 2^32-1`.
- For generated data require `2 <= n_samples <= 1000`, finite slope/intercept/noise/bounds, `0 <= noise_std <= 1e3`, `x_min < x_max`, and bounded generation parameters (absolute slope/intercept/bounds at most `1e6`).
- Cap request body size at 1 MiB. Enforce limits through Pydantic and service checks; reject rather than silently truncate. Error responses use a stable code, message, and optional field details; invalid requests return HTTP 422 and oversized bodies HTTP 413.
- Check every training iteration's intermediate predictions, residuals, gradients, candidate parameters, and loss for finiteness before proceeding. A non-finite intermediate or unstable update is a numerical-divergence error, never a successful response containing NaN/Infinity or a partial result.
- Return numerical divergence as a structured HTTP 422 error, for example `{"error":{"code":"NUMERICAL_DIVERGENCE","message":"Gradient descent produced a non-finite intermediate value.","details":{"step":12,"value":"gradient_slope"}}}`. Keep the code stable; details identify the step and intermediate field without including an invalid numeric payload.
- Input-valid requests can still diverge because of feature scale and learning rate. Detect divergence at the iteration where it first occurs. Whether to additionally reject risky learning-rate/data-scale combinations before computation must be settled during implementation.

For generated requests, the seed controls all random sampling/noise and is returned. Inline points are already deterministic; the seed is still echoed for a uniform contract. A repeated generated request with identical fields and seed returns the same dataset and equivalent model output within floating-point tolerance. No server session is required to continue training.

## 7. Non-functional requirements and measurable targets

Initial targets for Linear Regression V1:

- ML engine compute time: p95 <= 100 ms for 1,000 points and 500 gradient steps.
- Full FastAPI request time: p95 <= 250 ms for the same request, measured from request receipt through serialized response on the documented reference backend environment.
- Response payload: <= 1 MiB for every valid V1 request.
- Frontend interaction: p95 <= 1,000 ms from submission to updated chart after a successful response, on the documented local reference setup (API response included).

These are targets, not achieved results. Before claiming them, benchmark at least 100 warmed requests for the backend targets and 30 representative UI interactions for frontend response time; record hardware/runtime versions, workload, percentiles, and payload size in the development log. Do not weaken correctness or silently drop chart data to meet a target.

## 8. Mathematical acceptance and testing

- Verify gradients and one-step updates against hand-calculated fixtures; verify `loss_history` length and MSE values.
- Add explicit numerical-divergence tests that use finite, schema-valid inputs but cause a non-finite intermediate during gradient descent. Assert detection occurs at the first failing step, the API returns the structured `NUMERICAL_DIVERGENCE` error with step/field details, and no partial success response or NaN/Infinity is serialized. Also test that large but finite intermediates continue when they remain finite.
- Compare converged fitted parameters/predictions to an appropriate scikit-learn reference with documented tolerance; scikit-learn is a reference, not the source for the explicit GD step trace.
- Same seeded synthetic request is repeatable; invalid shapes, ranges, and non-finite inputs fail clearly.
- Tests cover frontend state/API behavior (Vitest + React Testing Library), backend schemas/routes/engine (pytest), and frontend-to-API contracts.
- Later algorithms meet their dimension-specific mathematical acceptance checks and chart/result consistency checks.

## 9. Assumptions and open decisions

- V1 uses simple batch gradient descent without regularization; feature scaling is not implicit. Large feature magnitudes can require a small learning rate, so instability must be detected and explained.
- Decide whether the synthetic data request includes an optional distribution/seeded shuffle and whether response echoes the complete effective request.
- Select static frontend and Python backend hosting, define CORS origins, health checks, and environment variables before deployment.
- Performance targets must be revisited on the actual deployment tier after benchmarks; no benchmark has been run as part of this documentation update.
