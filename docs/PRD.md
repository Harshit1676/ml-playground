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

## 3. Public launch scope

Algorithms ship in this order:

1. **Linear Regression:** V1 uses one numeric input feature `x` and one continuous target `y`. Fit a line, show predictions/residual error, and expose explicit gradient descent steps.
2. **Logistic Regression:** the current lab uses one numeric input feature `x` and a binary label `y`. Show probabilities, thresholded classes, classification metrics, and training loss. A two-feature decision-boundary experience is a future extension.
3. **K-Means:** two numeric input features, no labels. Show cluster assignments and centroids; explain initialization sensitivity.

**Later release:** KNN will use two numeric input features and class labels, with regions corresponding to selected `k` and distance rule. It is not in the public launch target.

Shared capabilities include reproducible datasets, bounded parameter controls, mathematically grounded charts, plain-language explanations, typed REST interactions, useful validation errors, responsive layout, and keyboard accessibility.

Out of scope for launch: KNN, the two-feature Logistic Regression decision boundary, AI chatbot/LLM, authentication, database or saved experiment history, Redis, Celery/task queues, arbitrary code execution, and other algorithms.

**Milestones 4A and 4B:** the backend and current learner lab implement a one-feature binary classifier (`x` plus label `y`). The lab shows a sigmoid probability curve, observed labels, a threshold guide, classification metrics, and binary cross-entropy history; it does not show a two-dimensional decision boundary. A later extension must expand the API and engine to two features before claiming that decision-boundary experience. That extension will use a vector score `z = w1*x1 + w2*x2 + intercept`, preserve the binary loss/threshold/metric semantics, and provide response data sufficient to render the decision boundary; the future request/response field layout remains to be designed.

## 4. Core experience

1. Learner chooses an algorithm, starting with Linear Regression.
2. Page presents data, controls, visualization, and explanation.
3. Learner changes data or model parameters and advances a learning step where available.
4. Frontend sends a typed request; FastAPI validates and calls the Python ML engine.
5. Frontend updates the plot and explanation from the response for that exact request.
6. Reset restores defaults and seed so the baseline can be reproduced.

## 5. Functional requirements

- **FR-1 Progression:** launch Linear Regression, then Logistic Regression, then K-Means; plan KNN for a later release.
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
- For generated data require `2 <= n_samples <= 1000`, finite slope/intercept/noise/bounds, `0 <= noise_std <= 1e3`, `x_min < x_max`, and bounded generation parameters (absolute slope/intercept/bounds at most `1e6`). Generated observations are subject to the same `abs(x) <= 1e6` and `abs(y) <= 1e6` limits as inline points. If finite generated `y` values exceed the limit, reject the request with HTTP 422 and structured `DATASET_VALUE_OUT_OF_RANGE` details identifying the field/index and allowed magnitude; do not clip or rescale. Reserve `NUMERICAL_DIVERGENCE` for non-finite arithmetic.
- Cap request body size at 1 MiB. Enforce limits through Pydantic and service checks; reject rather than silently truncate. Error responses use a stable code, message, and optional field details; invalid requests return HTTP 422 and oversized bodies HTTP 413.
- Check every training iteration's intermediate predictions, residuals, gradients, candidate parameters, and loss for finiteness before proceeding. Non-finite arithmetic is a numerical-divergence error, never a successful response containing NaN/Infinity or a partial result. Finite generated observations outside the documented dataset magnitude limit are ordinary validation failures, not numerical divergence.
- Return numerical divergence as a structured HTTP 422 error, for example `{"error":{"code":"NUMERICAL_DIVERGENCE","message":"Gradient descent produced a non-finite intermediate value.","details":{"step":12,"value":"gradient_slope"}}}`. Keep the code stable; details identify the step and intermediate field without including an invalid numeric payload.
- Input-valid requests can still diverge because of feature scale and learning rate. Detect divergence at the iteration where it first occurs. Whether to additionally reject risky learning-rate/data-scale combinations before computation must be settled during implementation.

For generated requests, the seed controls all random sampling/noise and is returned. Inline points are already deterministic; the seed is still echoed for a uniform contract. A repeated generated request with identical fields and seed returns the same dataset and equivalent model output within floating-point tolerance. No server session is required to continue training.

## 7. Logistic Regression backend contract (Milestone 4A)

### Endpoint and model

`POST /api/v1/experiments/logistic-regression` trains a univariate binary Logistic Regression model with `z = coefficient * x + intercept` and `p = sigmoid(z)`. It uses mean binary cross-entropy and explicit full-batch gradient descent without regularization. Each update computes both gradients from the same pre-update parameters and applies them simultaneously. The response includes the initial loss followed by the loss after every update (`steps + 1` values). The endpoint is stateless.

### Request example

```json
{
  "dataset": {"kind": "points", "x": [-2.0, -1.0, 1.0, 2.0], "y": [0, 0, 1, 1]},
  "parameters": {"initial_coefficient": 0.0, "initial_intercept": 0.0, "learning_rate": 0.1, "steps": 0},
  "threshold": 0.5,
  "seed": 42
}
```

The dataset may instead be `{"kind":"synthetic_logistic","n_samples":100,"true_coefficient":1.5,"true_intercept":0.0,"x_min":-4.0,"x_max":4.0}`. A local NumPy generator seeded by `seed` samples `x` uniformly from `[x_min, x_max)`, then samples each binary label from a Bernoulli distribution with probability `sigmoid(true_coefficient*x + true_intercept)`. Seeded requests are repeatable. The seed is echoed for inline points as well, although those observations do not use it.

For `z_i = coefficient*x_i + intercept`, `p_i = sigmoid(z_i)`, and `n` observations, the objective is `BCE = (1/n) * sum(logaddexp(0, z_i) - y_i*z_i)`. The gradients are `d_coefficient = mean((p-y)*x)` and `d_intercept = mean(p-y)`. Each update subtracts `learning_rate * gradient` from both parameters using the same pre-update values. `steps: 0` evaluates the initial model without updating it.

Example response for the request above:

```json
{
  "algorithm": "logistic_regression",
  "seed": 42,
  "dataset": {"x": [-2.0, -1.0, 1.0, 2.0], "y": [0, 0, 1, 1]},
  "parameters": {"initial_coefficient": 0.0, "initial_intercept": 0.0, "learning_rate": 0.1, "steps": 0},
  "threshold": 0.5,
  "model": {"coefficient": 0.0, "intercept": 0.0},
  "predictions": {"probabilities": [0.5, 0.5, 0.5, 0.5], "predicted_classes": [1, 1, 1, 1]},
  "metrics": {
    "binary_cross_entropy": 0.6931471805599453,
    "confusion_matrix": {"true_positive": 2, "true_negative": 0, "false_positive": 2, "false_negative": 0},
    "accuracy": 0.5,
    "precision": 0.5,
    "recall": 1.0,
    "f1": 0.6666666666666666
  },
  "training": {"steps_completed": 0, "loss_history": [0.6931471805599453]}
}
```

### Validation and response semantics

- Inline `x` and `y` arrays have equal lengths from 1 to 1,000; labels are exactly `0` or `1`. Synthetic sample count is 2 to 1,000. Feature values and coefficients/intercepts are finite with absolute magnitude at most `1e6`; synthetic `x_min` must be less than `x_max`.
- `0 < learning_rate <= 1`; `0 <= steps <= 500`; `0 <= threshold <= 1`; seed is an unsigned 32-bit integer. Request bodies are limited to 1 MiB. Invalid input returns the standard structured HTTP 422 error; oversized bodies return HTTP 413.
- Synthetic generation can probabilistically produce only one class, including for a fixed seed. Such a request returns structured HTTP 422 `SINGLE_CLASS_DATASET` with the observed class and sample count. It is not silently resampled or altered. This means not every seed is accepted, especially for small samples or extreme probabilities; repeatedly changing seeds until both classes appear conditions the selected sample and can bias what a learner sees. Inline one-class datasets are accepted so degenerate metric behavior is defined.
- Stable sigmoid branches avoid unnecessary exponential overflow; BCE is evaluated with `logaddexp(0, z) - y*z`. Non-finite intermediate arithmetic returns structured HTTP 422 `NUMERICAL_DIVERGENCE` with step and field details, without a partial result or non-finite JSON.
- Classification uses `probability >= threshold` for class 1. Confusion-matrix counts sum to the observation count. Accuracy, precision, recall, and F1 are returned with values in `[0,1]`; a zero denominator yields `0.0` for the affected metric.
- The response contains algorithm, seed, effective dataset, training parameters, threshold, final coefficient/intercept, per-observation probabilities and predicted classes, binary cross-entropy, confusion matrix, accuracy/precision/recall/F1, completed steps, and loss history. Response validation checks array dimensions, threshold-derived classes, confusion counts, and `steps + 1` loss entries.

The initial API intentionally does not return coefficient/intercept history. If the future UI requires a learning-step animation, a follow-up contract can add an explicit sequence of parameter pairs for each update; it must not infer parameters from loss values. Scikit-learn's default Logistic Regression includes regularization and solver behavior that differs from this unregularized explicit gradient descent, so it is not treated as an exact step-by-step oracle. Compare only a deliberately configured converged reference if a suitable like-for-like fixture is established.

## K-Means backend contract (Milestone 5A)

`POST /api/v1/experiments/k-means` accepts either inline points or a synthetic recipe. For `dataset.kind = "points"`, supply an ordered `dataset.points` array of 1–1,000 unlabeled coordinate pairs. Each coordinate must be a finite JSON number in `[-1e6, 1e6]`. Required integer fields are `parameters.k` (1–16 and at most the observation count), `parameters.max_iterations` (1–100) and a uint32 `seed`. The shared 1 MiB request limit and structured validation errors apply.

For `dataset.kind = "synthetic_clusters"`, supply integer `n_samples` (1–1,000), `centers` (1–16 coordinate pairs), finite `spread` (0–1e6) and uint32 `generation_seed`. Each observation chooses a source center uniformly with replacement and adds independent uniform offsets on both axes from `-spread` to `+spread`. Zero spread produces the selected source coordinates exactly. Every source square must fit inside the coordinate bounds, or the API returns HTTP 422. The cluster count requested for fitting is independent of the number of source centers. The response echoes the recipe alongside `dataset.points`, the effective observations passed to training. Only the generation RNG uses `generation_seed`; the top-level seed selects the initial centroids.

The seed selects observation rows without replacement. Cluster IDs are zero-based positions in that sample. Repeating the same ordered input and seed reproduces the run. Duplicate coordinates are valid. Equal distances go to the lowest cluster ID, and an empty cluster retains its previous center. The result can therefore contain fewer than `k` occupied clusters.

Each iteration takes the mean of the points currently assigned to each occupied center, then assigns every point to its nearest updated center. Stable assignments mark convergence. At the iteration limit, assignments and inertia still describe the returned centers. If assignments changed on that final update, another mean update could move the centers again.

The response echoes the dataset, parameters and seed. It includes `model.centroids`, per-point `assignments` in input order, `metrics.inertia`, and a `training` summary containing `initial_centroids`, `iterations_completed`, `converged`, `inertia_history` and `snapshots`. Snapshot 0 captures the sampled centroids and their nearest-point assignments. Each following snapshot contains its `iteration`, updated `centroids`, nearest-centroid `assignments` in input order, and `inertia` computed for that same state. There are at most 101 snapshots. The last snapshot supplies the final model, assignments and metric.

Sampling distinct rows can select repeated coordinates even when the data contains at least `k` distinct locations. Cluster counts at any iteration are the frequencies of IDs 0 through `k-1` in that snapshot. Stable assignments can include zero counts. On the last permitted update, stable assignments give `converged: true`; changed assignments give `converged: false`.

## 8. Non-functional requirements and measurable targets

Initial targets for Linear Regression V1:

- ML engine compute time: p95 <= 100 ms for 1,000 points and 500 gradient steps.
- Full FastAPI request time: p95 <= 250 ms for the same request, measured from request receipt through serialized response on the documented reference backend environment.
- Response payload: <= 1 MiB for every valid V1 request.
- Frontend interaction: p95 <= 1,000 ms from submission to updated chart after a successful response, on the documented local reference setup (API response included).

These are targets, not achieved results. Before claiming them, benchmark at least 100 warmed requests for the backend targets and 30 representative UI interactions for frontend response time; record hardware/runtime versions, workload, percentiles, and payload size in the development log. Do not weaken correctness or silently drop chart data to meet a target.

## 9. Mathematical acceptance and testing

- Verify gradients and one-step updates against hand-calculated fixtures; verify `loss_history` length and MSE values.
- Add explicit numerical-divergence tests that use finite, schema-valid inputs but cause a non-finite intermediate during gradient descent. Assert detection occurs at the first failing step, the API returns the structured `NUMERICAL_DIVERGENCE` error with step/field details, and no partial success response or NaN/Infinity is serialized. Also test that large but finite intermediates continue when they remain finite.
- Compare converged fitted parameters/predictions to an appropriate scikit-learn reference with documented tolerance; scikit-learn is a reference, not the source for the explicit GD step trace.
- Same seeded synthetic request is repeatable; invalid shapes, ranges, and non-finite inputs fail clearly.
- Tests cover frontend state/API behavior (Vitest + React Testing Library), backend schemas/routes/engine (pytest), and frontend-to-API contracts.
- Later algorithms meet their dimension-specific mathematical acceptance checks and chart/result consistency checks.

## 10. Assumptions and open decisions

- V1 uses simple batch gradient descent without regularization; feature scaling is not implicit. Large feature magnitudes can require a small learning rate, so instability must be detected and explained.
- Decide whether the synthetic data request includes an optional distribution/seeded shuffle and whether response echoes the complete effective request.
- Select static frontend and Python backend hosting, define CORS origins, health checks, and environment variables before deployment.
- Performance targets must be revisited on the actual deployment tier after benchmarks; no benchmark has been run as part of this documentation update.
