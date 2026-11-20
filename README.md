# ML Playground

ML Playground is an interactive platform for beginner-to-intermediate learners to explore foundational machine learning through data, controls, visualizations, and explanations.

## MVP

The public launch target is Linear Regression, Logistic Regression, then K-Means. The current Logistic Regression lab uses one feature and binary labels; a two-feature decision-boundary experience is a future extension. K-Nearest Neighbors (KNN) is planned for a later release. Learners adjust datasets and parameters, inspect mathematically grounded results, and follow learning-step calculations where supported. Experiments are reproducible with explicit random seeds.

The frontend owns controls, interaction, visualization, and explanation presentation. ML computation runs in a separate Python engine behind a stateless FastAPI service. MVP excludes authentication, a database, Redis, Celery, and an AI chatbot.

## Final stack

- **Frontend:** React, Vite, TypeScript, Tailwind CSS, Plotly.js
- **Backend:** Python 3.11+, FastAPI, Pydantic, NumPy, scikit-learn
- **Testing:** Vitest, React Testing Library, pytest, and API integration tests
- **Deployment:** Static frontend hosting and a separate FastAPI service

## Monorepo

```text
frontend/   React + Vite app, controls, UI state, visualization, explanations
backend/    FastAPI routes/schemas/services and Python ML engine
docs/       Product, architecture, and development documentation
```

The ML engine handles dataset generation, training, predictions, metrics, and learning-step calculations; it does not run in React. See [the PRD](docs/PRD.md), [the architecture](docs/ARCHITECTURE.md), and [the development log](docs/DEVELOPMENT_LOG.md). Contributor workflow is in [AGENTS.md](AGENTS.md).

## Local development (Windows PowerShell)

Use two PowerShell terminals from the repository root.

Use Node.js **24.22.3 LTS**, pinned in the repository-root `.nvmrc` and constrained by `frontend/package.json`. This patch release is within the installed Vite, Vitest, and jsdom engine ranges. With nvm-windows, run `nvm use 24.22.3`; otherwise install Node.js 24.22.3 using the official Node.js installer. The current workstation's Node.js 25.2.0 is outside the supported range and may emit engine warnings; CI uses the pinned LTS version.

### Backend

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Copy the example values into the process environment as needed.
$env:APP_ENV = "development"
$env:CORS_ALLOWED_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The health endpoint is `http://localhost:8000/api/v1/health`. Interactive API documentation is at `http://localhost:8000/docs`.

To run backend tests in a separate terminal:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```

### Frontend

```powershell
cd frontend
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm ci
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://localhost:5173`). The page checks the backend health endpoint and offers a retry when it cannot connect. Frontend checks:

```powershell
npm test
npm run build
```

`frontend/.env.example` sets `VITE_API_BASE_URL`; Vite exposes this value to the browser bundle, so only public configuration belongs there. Do not put secrets in `VITE_*` variables.

`backend/.env.example` documents backend configuration, but the application reads operating-system environment variables directly and does not automatically load `.env` files. The PowerShell assignments above set them for the current terminal. In production, configure `APP_ENV=production` and `CORS_ALLOWED_ORIGINS` in the backend hosting platform's environment settings. Production startup fails if that origin list is missing or contains `*`.

## CI

`.github/workflows/ci.yml` runs on pull requests and pushes to `main`. It uses the Node version in `.nvmrc`, runs `npm ci`, `npm test`, and `npm run build` from `frontend/`, then installs `backend/requirements.txt` with Python 3.11 and runs pytest. These checks need no repository secrets. This is CI only; no deployment workflow is configured.

## Production deployment configuration

No hosting provider has been selected and nothing has been deployed.

- **Static frontend:** run `npm run build` from `frontend/`; upload the generated `frontend/dist/` directory to a static host. Set `VITE_API_BASE_URL` to the public FastAPI base URL in the frontend host's build-time environment. It is embedded in public assets and must not contain secrets.
- **FastAPI service:** install `backend/requirements.txt`, set `APP_ENV=production`, and set `CORS_ALLOWED_ORIGINS` to a comma-separated list of exact frontend origins, for example `https://playground.example.com`. Set these as backend service environment variables; do not use wildcard origins. Start with `python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"`, where the platform supplies `PORT`.
- **CORS methods and preflight:** the API already allows `GET`, `POST`, and `OPTIONS` for the health and Linear Regression routes. Production preflight must use an explicitly allowed HTTPS origin; `content-type` is permitted. When adding other HTTP methods or headers, enable and test only those required.

### Deployment smoke-test checklist

1. Confirm the static host serves the frontend build output and the configured `VITE_API_BASE_URL` points to the deployed API.
2. Confirm the backend starts on the platform-provided port with `APP_ENV=production` and explicit `CORS_ALLOWED_ORIGINS`.
3. Request `/api/v1/health` directly and confirm HTTP 200 with `{"status":"ok"}`.
4. Load the frontend from its real origin and confirm it reports the backend as connected.
5. Confirm the allowed production origin can preflight the Linear Regression `POST` with `content-type`, and an unlisted origin receives no `Access-Control-Allow-Origin` header.
6. Verify no secrets are present in frontend build assets or client-visible `VITE_*` settings.

These instructions describe the deployment contract, not a provider-specific release setup. Production domain, TLS, DNS, monitoring, and platform-specific health-check configuration remain open.

## Important files and dependencies

- `frontend/src/App.tsx` provides query-string navigation between the Linear Regression, Logistic Regression and K-Means labs and reports backend loading/connected/failed states.
- `frontend/src/lib/api/health.ts` provides the typed health client, validates the response shape, and uses `VITE_API_BASE_URL` (defaulting to `http://localhost:8000`).
- `frontend/src/styles.css` imports Tailwind CSS and contains responsive experiment controls, results, and chart styling.
- `frontend/vite.config.ts` configures the React and Tailwind Vite plugins and Vitest's jsdom environment.
- `frontend/src/features/linear-regression/LinearRegressionExperience.tsx` and `frontend/src/features/logistic-regression/LogisticRegressionExperience.tsx` own each lab's form and page state; each feature's `request.ts` validates controls and constructs its existing API payload.
- `frontend/src/lib/api/linearRegression.ts` sends typed experiment requests and validates successful response shape or surfaces the backend's structured error envelope.
- `frontend/src/lib/api/logisticRegression.ts` sends the existing Logistic Regression request and validates response semantics, including returned probabilities, threshold metrics, confusion counts, and loss history.
- `frontend/src/features/linear-regression/RegressionCharts.tsx` displays response-derived result charts; `frontend/src/lib/visualization/linearRegression.ts` maps API observations, final model parameters, and returned loss history into Plotly traces.
- `frontend/src/features/logistic-regression/LogisticRegressionCharts.tsx` displays the observed binary labels, sigmoid derived from returned model parameters, active threshold, and returned BCE history; `frontend/src/features/logistic-regression/classification.ts` computes threshold-dependent labels and metrics locally.
- `frontend/src/components/PlotlyFigure.tsx` loads the React Plotly renderer only when result charts are shown. `react-plotly.js` adapts Plotly to React, `plotly.js-basic-dist-min` supplies the smaller runtime chart bundle, and `plotly.js-dist-min` supplies development-only TypeScript definitions.
- `frontend/src/App.test.tsx`, feature tests, API-client tests, and chart-transform tests cover algorithm navigation, UI states, payloads, errors, stale requests, and response-derived visuals.
- `backend/app/main.py` creates FastAPI and reads `APP_ENV`/`CORS_ALLOWED_ORIGINS` from the process environment; it defaults to Vite localhost origins in development and requires explicit non-wildcard origins in production. `backend/app/api/routes/health.py` defines the route and Pydantic response model.
- `backend/app/api/router.py` mounts versioned routes under `/api/v1`; `backend/tests/test_health.py` covers the health response and CORS.
- `backend/app/api/routes/logistic_regression.py`, `backend/app/schemas/logistic_regression.py`, `backend/app/services/logistic_regression.py`, and `backend/app/ml/logistic_regression.py` separate the Logistic Regression endpoint, typed contract, stateless orchestration/metrics, and NumPy training engine.
- `.nvmrc` pins Node.js 24.22.3 for local use and GitHub Actions; `.github/workflows/ci.yml` runs frontend and backend checks without secrets.
- `frontend/package.json` declares React, Vite, TypeScript, Tailwind CSS, Plotly's basic partial bundle, and the React Plotly adapter. `plotly.js-dist-min` is retained as a development-only type source. Tailwind's Vite plugin integrates utility generation; Vitest runs the tests, React Testing Library renders components in jsdom, and pytest plus HTTPX exercises FastAPI in-process.
- `backend/requirements.txt` bounds FastAPI/Pydantic for the API, Uvicorn for local serving, and HTTPX/pytest for tests.

## Linear Regression API (Milestone 2)

`POST /api/v1/experiments/linear-regression` accepts either inline points (`x` and `y` arrays) or a `synthetic_linear` dataset and returns the effective dataset, seed, initial configuration, fitted slope/intercept, predictions, MSE, completed update count, and loss history. The endpoint is stateless: to continue training, send the same data/configuration with the previous model parameters as the next initial values and `steps: 1`.

Example request (the three point fixture used by tests):

```json
{
  "dataset": {"kind": "points", "x": [-1, 0, 1], "y": [-1, 1, 3]},
  "parameters": {"initial_slope": 0, "initial_intercept": 0, "learning_rate": 0.1, "steps": 1},
  "seed": 42
}
```

The response contains `algorithm: "linear_regression"`, `dataset`, `seed`, `parameters`, `model`, `predictions.y_hat`, `metrics.mse`, and `training` with `steps_completed` and `loss_history`. Here, `MSE = mean((slope*x + intercept - y)^2)`. Each batch step calculates `d_slope = 2*mean((prediction-y)*x)` and `d_intercept = 2*mean(prediction-y)` from the same pre-update parameters, then subtracts `learning_rate * gradient` from both parameters simultaneously. Loss history starts with the initial MSE and includes each post-update MSE, so it always contains `steps + 1` finite values. With the example fixture, one update yields slope `4/15`, intercept `0.2`, and MSE approximately `2.64296296296`.

Example response for the fixture above:

```json
{
  "algorithm": "linear_regression",
  "seed": 42,
  "dataset": {"x": [-1.0, 0.0, 1.0], "y": [-1.0, 1.0, 3.0]},
  "parameters": {"initial_slope": 0.0, "initial_intercept": 0.0, "learning_rate": 0.1, "steps": 1},
  "model": {"slope": 0.26666666666666666, "intercept": 0.2},
  "predictions": {"y_hat": [-0.06666666666666665, 0.2, 0.4666666666666667]},
  "metrics": {"mse": 2.642962962962963},
  "training": {"steps_completed": 1, "loss_history": [3.6666666666666665, 2.642962962962963]}
}
```

Inline data accepts 1–1,000 observations; generated data accepts 2–1,000. Both inline and generated observations must have finite `x` and `y` values with absolute magnitude at most `1e6`. Synthetic configuration values are bounded separately; if they produce a finite `y` outside the observation limit, the request returns HTTP 422 with `DATASET_VALUE_OUT_OF_RANGE`, field/index and limit details. Values are never clipped or rescaled. Initial parameters are finite and bounded to absolute value `1e6`; synthetic noise standard deviation is 0–1,000; learning rate is `(0, 1]`; steps are 0–500; seed is an unsigned 32-bit integer; and request bodies are limited to 1 MiB. Other invalid input returns HTTP 422 with `error.code: "VALIDATION_ERROR"`; oversized bodies return HTTP 413 with `REQUEST_TOO_LARGE`. Non-finite intermediate predictions, residuals, gradients, candidate parameters, or losses return HTTP 422 with `NUMERICAL_DIVERGENCE`, the first failing step and field, and no partial result. A finite but increasing loss is returned as calculated; the API does not silently rescale data or alter the requested learning rate.

## Backend module responsibilities

- `backend/app/api/routes/experiments.py` exposes the thin versioned HTTP route.
- `backend/app/schemas/linear_regression.py` defines discriminated input schemas, bounds, and response consistency validation; `schemas/errors.py` describes structured API errors.
- `backend/app/services/linear_regression.py` resolves inline/generated datasets and assembles the API response without retaining state.
- `backend/app/ml/linear_regression.py` is the FastAPI-independent NumPy engine: seeded data generation, predictions, MSE, gradients, updates, and divergence detection.
- `backend/app/middleware.py` enforces the 1 MiB write-body limit before JSON parsing.
- `backend/tests/test_linear_regression.py` covers mathematical fixtures, determinism, continuation, validation, numerical safety, API behavior, and a scikit-learn comparison.

NumPy supplies explicit float64 vector calculations and makes each educational gradient step inspectable. scikit-learn's `LinearRegression` is used only as a converged-fit reference in tests; for its seeded 100-point fixture, both slope and intercept are compared with absolute tolerance `1e-8`. It does not replace the requested batch-gradient-descent trace. New backend runtime dependencies are NumPy and scikit-learn; FastAPI/Pydantic provide HTTP/schema validation, while pytest/HTTPX provide route tests. The frontend consumes this existing stateless API; it does not run training calculations in React.

## Linear Regression learning experience (Milestone 3)

Open the Vite frontend using the commands in [Local development](#local-development-windows-powershell), ensure the FastAPI service is running, then choose a seeded synthetic dataset or enter comma-separated inline `x` and `y` observations. Adjust the sample count, true line, noise, feature range, seed, starting slope/intercept, learning rate, and update count. The numeric controls are validated against the existing backend schema before a request is sent. Synthetic observations that exceed the backend's documented magnitude limit are reported as structured validation errors; the frontend does not clip or rescale them.

Run the experiment to send `POST /api/v1/experiments/linear-regression` with the documented `dataset`, `parameters`, and `seed` request fields. The typed API client validates the returned response shape. The results show the returned final slope, intercept, MSE, and completed steps. The scatterplot uses the returned observations and returned final model parameters. The loss chart plots the returned `training.loss_history` at step indices from zero through the completed step count; the frontend never invents intermediate parameter values. Network failures, backend errors, loading, and retryable failures have separate UI states. Submitting a newer request or invalid settings cancels/invalidate older responses so they cannot replace the current state.

Plotly is dynamically loaded when experiment results are displayed, keeping it out of the initial application chunk. The app uses the official `plotly.js-basic-dist-min` partial bundle, which includes the scatter trace used for observations, the fitted line, and the loss curve. The latest verified build emitted a 1,160.44 kB minified (384.74 kB gzip) deferred chart chunk; Vite still reports its minified-size warning. The API remains stateless, and this milestone adds no backend routes or algorithms.

## Logistic Regression backend (Milestone 4A)

The backend adds `POST /api/v1/experiments/logistic-regression` for a one-feature binary classifier. The M4B learner experience uses this contract as implemented. A two-feature classifier and a two-dimensional decision-boundary view remain future extensions and are not part of this milestone.

Request example:

```json
{
  "dataset": {"kind": "points", "x": [-2, -1, 1, 2], "y": [0, 0, 1, 1]},
  "parameters": {"initial_coefficient": 0, "initial_intercept": 0, "learning_rate": 0.1, "steps": 10},
  "threshold": 0.5,
  "seed": 42
}
```

The dataset may also use `kind: "synthetic_logistic"` with `n_samples`, `true_coefficient`, `true_intercept`, `x_min`, and `x_max`. A local NumPy generator samples features uniformly from `[x_min, x_max)` and labels from Bernoulli probabilities produced by the sigmoid of the configured true linear score. A draw can round to the upper endpoint when represented as a float64. Repeating a synthetic request with the same seed reproduces the data. Inline points are returned as submitted; the seed is echoed but does not generate them.

Training uses `z = coefficient*x + intercept`, a numerically stable sigmoid, mean binary cross-entropy, and explicit full-batch gradient descent with simultaneous parameter updates. The response includes the effective dataset, model coefficient/intercept, per-observation probabilities and predicted classes, threshold, confusion matrix, accuracy/precision/recall/F1, and loss history with the initial loss plus each update. Class 1 is selected when `probability >= threshold`; metrics with zero denominators return `0.0`. Synthetic requests that produce only one class return structured HTTP 422 `SINGLE_CLASS_DATASET`; the generator does not silently resample. Retrying seeds until both classes appear selects samples conditionally and can bias learner expectations, particularly with small datasets or extreme probabilities. Non-finite intermediate arithmetic returns `NUMERICAL_DIVERGENCE` through the standard error envelope. Request limits are at most 1,000 observations, 500 updates, 1 MiB request body, finite feature/parameter magnitude up to `1e6`, learning rate `(0,1]`, threshold `[0,1]`, and uint32 seed.

No coefficient/intercept history is currently returned. A future learning animation should add explicit parameter snapshots to the API instead of reconstructing them from loss values. Scikit-learn's regularized default Logistic Regression is not an exact reference for this unregularized explicit gradient-descent trace.

## Logistic Regression learning experience (Milestone 4B)

Use the algorithm selector to open Logistic Regression directly or follow the `?algorithm=logistic-regression` deep link. The selector works with keyboard input and marks the active lab. Switching algorithms keeps each lab mounted so its form and result state remain available; **Reset to defaults** clears the Logistic Regression run and restores the seeded synthetic configuration.

Choose seeded synthetic data or edit comma-separated inline `x` values and binary `y` labels. The synthetic form exposes sample count, true coefficient/intercept, feature range, and seed. The seed control is hidden for inline observations; the required API seed remains in the request using the default seed. Configure the initial coefficient/intercept, learning rate, number of updates, and submitted classification threshold. Controls are validated against the current FastAPI schema before sending `POST /api/v1/experiments/logistic-regression`.

The response shows the learned coefficient and intercept, backend BCE, training updates, and classification metrics. The probability chart places observed labels at 0/1 and derives a sorted sigmoid curve from the returned final parameters; the loss chart plots only the returned BCE history, including step zero. There is no coefficient/intercept history, so the frontend does not invent training animation values.

After training, move the exploration threshold to immediately recompute predicted classes, the confusion matrix, accuracy, precision, recall, and F1 from the returned probabilities and actual labels. Classification uses `probability >= threshold`; an undefined zero-denominator metric is shown as 0%, matching the backend. The submitted threshold remains visible separately. Threshold exploration does not retrain or change the model, probabilities, BCE, or loss history. The confusion matrix labels rows as actual classes and columns as predicted classes, with TP/TN/FP/FN definitions. Accuracy is presented alongside precision and recall because it is not a complete measure of model quality.

Synthetic generation can return the structured `SINGLE_CLASS_DATASET` error for some seeds and settings; the UI explains that changing the seed or generator settings may help. Inline single-class observations remain valid and follow the backend's zero-denominator metric convention. This milestone remains one-feature only, has no training animation or two-dimensional decision boundary, and requires the FastAPI service to be available for experiments.

Local verification commands from `frontend/` are `npm test` and `npm run build`; from `backend/` run `python -m pytest`. The M4B automated integration tests mock the HTTP response in the browser component. A live browser run or production deployment is not implied by passing these tests.

## Linear Regression manual browser verification checklist

1. Run the seeded default experiment and confirm the returned observations, fitted line, result metrics, and loss history appear.
2. Change the learning rate, run again, and confirm the returned model/loss values update. Change the step count and confirm the loss chart contains `steps + 1` points.
3. Switch to inline points: confirm the seed control is hidden, edit `x`/`y`, and confirm the request still includes the API-required seed field. Switch back to synthetic data and confirm the seed control returns.
4. Click **Reset to defaults** and confirm the synthetic seed, noise `1.2`, learning rate, steps, and result state reset.
5. Enter an invalid value (for example, learning rate `0`) and confirm a useful field error appears and no invalid request is sent. Try a finite synthetic configuration that generates an out-of-range target and confirm the structured backend error is shown.
6. Stop FastAPI while the page is open or point `VITE_API_BASE_URL` to an unavailable local port, then confirm backend-unavailable and retry states.
7. Use browser device emulation at about 390 px wide and confirm controls/results stack without horizontal overflow.
8. Clear DevTools Console, reload, run experiments, switch datasets, and reset; confirm no JavaScript errors appear.

## Logistic Regression manual browser verification checklist

1. Open Logistic Regression from the selector and load `/?algorithm=logistic-regression` directly; confirm the active item and backend health status.
2. Run the default synthetic experiment and compare the probability chart, classification metrics, confusion matrix, and BCE chart with the returned API response.
3. Move the exploration threshold, including an exact probability boundary; confirm labels, confusion counts, and threshold metrics update without a second POST. Confirm the submitted threshold, learned coefficients, probabilities, BCE, and loss history remain unchanged.
4. Switch to inline observations, confirm the seed control is hidden, edit paired finite features and binary labels, and run. Switch back and verify the synthetic seed/configuration is preserved.
5. Try mismatched arrays, a non-binary label, out-of-range controls, and the generated single-class case; verify actionable field or structured API errors and that previous results are cleared for invalid submissions.
6. Reset after a completed run and confirm defaults are restored and old metrics/charts disappear. Switch between algorithm labs and use browser back/forward to verify navigation state.
7. Stop FastAPI and confirm the health and experiment failure states are clear; then use a narrow viewport around 390 px and check for horizontal overflow and JavaScript console errors.

## K-Means backend (Milestone 5A)

For the interactive frontend and manual checks, see [K-Means learning experience](#k-means-learning-experience-milestone-5b).

Use `training.snapshots[0]` to inspect initialization. Each subsequent entry contains the centroids, assignments and inertia after one completed update. The last entry matches the final summary.

Open `/docs` on the running FastAPI service and try `POST /api/v1/experiments/k-means` with four unlabeled 2D observations.

```json
{
  "dataset": {"kind": "points", "points": [[0, 0], [2, 4], [8, 8], [10, 12]]},
  "parameters": {"k": 2, "max_iterations": 10},
  "seed": 42
}
```

For this example the sampled centers start at `[[0, 0], [10, 12]]`. After one update the centers are `[[1, 2], [9, 10]]` and the assignments are `[0, 0, 1, 1]`. Each point now contributes 5 to the sum of squared distances, so inertia falls from 40 to 20. Changing the seed explores the effect of starting from different observations. See the [K-Means contract](docs/PRD.md#k-means-backend-contract-milestone-5a) for limits and stopping rules.

The complete response for this request is

```json
{
  "algorithm": "kmeans", "seed": 42, "parameters": {"k": 2, "max_iterations": 10},
  "dataset": {"kind": "points", "points": [[0,0],[2,4],[8,8],[10,12]]},
  "model": {"centroids": [[1,2],[9,10]]}, "assignments": [0,0,1,1], "metrics": {"inertia": 20},
  "training": {
    "initial_centroids": [[0,0],[10,12]], "iterations_completed": 1, "converged": true,
    "inertia_history": [40,20],
    "snapshots": [
      {"iteration": 0, "centroids": [[0,0],[10,12]], "assignments": [0,0,1,1], "inertia": 40},
      {"iteration": 1, "centroids": [[1,2],[9,10]], "assignments": [0,0,1,1], "inertia": 20}
    ]
  }
}
```

To generate points, supply a cluster recipe to the same endpoint. Each point is drawn from a square centered on a randomly selected source center. Keep `generation_seed` fixed and change the top-level `seed` to try different initial centroids on the same observations.

```json
{
  "dataset": {"kind": "synthetic_clusters", "n_samples": 2, "centers": [[-4,2],[6,-3]], "spread": 0.25, "generation_seed": 1},
  "parameters": {"k": 2, "max_iterations": 10},
  "seed": 42
}
```

This small example gives each point its own centroid. The response includes every generated observation in training order.

```json
{
  "algorithm": "kmeans", "seed": 42, "parameters": {"k": 2, "max_iterations": 10},
  "dataset": {
    "kind": "synthetic_clusters", "n_samples": 2, "centers": [[-4,2],[6,-3]], "spread": 0.25, "generation_seed": 1,
    "points": [[-3.774768151837032,1.822079806359817],[6.224324723568622,-3.094084273994757]]
  },
  "model": {"centroids": [[-3.774768151837032,1.822079806359817],[6.224324723568622,-3.094084273994757]]},
  "assignments": [0,1], "metrics": {"inertia": 0},
  "training": {
    "initial_centroids": [[-3.774768151837032,1.822079806359817],[6.224324723568622,-3.094084273994757]],
    "iterations_completed": 1, "converged": true, "inertia_history": [0,0],
    "snapshots": [
      {"iteration": 0, "centroids": [[-3.774768151837032,1.822079806359817],[6.224324723568622,-3.094084273994757]], "assignments": [0,1], "inertia": 0},
      {"iteration": 1, "centroids": [[-3.774768151837032,1.822079806359817],[6.224324723568622,-3.094084273994757]], "assignments": [0,1], "inertia": 0}
    ]
  }
}
```

## K-Means learning experience (Milestone 5B)

Start FastAPI and Vite using the local development commands above. Open `http://localhost:5173/?algorithm=k-means` and choose **Run experiment**. The defaults draw 150 points around three source centers using spread 1.5 and generation seed 42. K-Means fits three centroids with initialization seed 7 and up to 30 updates.

Enter one `x,y` pair per line for source centers or inline observations. Source centers describe where synthetic points can be drawn. The fitted centroids are learned from the resulting observations. Keep the generation recipe fixed and change only the initialization seed to investigate how starting positions affect the grouping.

Results open at the last completed iteration. Use the slider, Previous, Next, Initialization or Final result to inspect the run. Small markers show observations and crosses show centroids. The membership table lists every cluster, including empty ones and centroids that coincide. Expand **Inspect individual assignments** to read the cluster ID for any observation. The second plot highlights the selected iteration on the returned inertia curve. Inertia measures within-cluster squared distance and typically favors larger values of `k`.

### Manual browser checks

1. Run the default synthetic experiment twice with different initialization seeds. In the network responses, compare the effective points and initial centroids.
2. Step from the final result back to initialization. Compare both plots and both membership views with the same numbered snapshot. Moving the slider should send zero experiment requests.
3. Switch to inline points. Enter `0,0` three times, followed by `1,0`, `3,0`, `9,0` and `10,0`, each on its own line. Set `k=4`, initialization seed 2 and one maximum iteration. Inspect the empty centers and the iteration-limit explanation. Increase the limit to 10 to see stable assignments.
4. Switch through all three labs, use browser Back and Forward, and return at mobile width. Check preserved inputs, the selected step, chart widths and keyboard navigation.
5. Submit an incomplete coordinate pair after a successful run, then reset. Disconnect the backend, run again, restore connectivity and retry. Check the console and the experiment request count.
