# Development Log

## 2026-11-19 — Milestone 2 review fixes

- Clarified that the PRD's absolute `1e6` observation bound applies to generated `x` and `y` as well as inline values. Synthetic configurations that produce a finite out-of-range `y` now return structured HTTP 422 `DATASET_VALUE_OUT_OF_RANGE` with field, index, actual value, and maximum magnitude; values are not clipped/rescaled and `NUMERICAL_DIVERGENCE` remains reserved for non-finite arithmetic.
- Expanded production CORS coverage for allowed HTTPS POST preflight, unlisted origins, and the `content-type` header. Updated the README to reflect that POST is already enabled.
- Added tests for valid magnitude boundaries, synthetic generator parameter boundaries, exact 1 MiB bodies, oversized chunked bodies without Content-Length, malformed JSON error envelopes, and finite successful JSON responses.
- Verification: backend pytest passed (41 tests); frontend `npm test` passed (6 tests); frontend `npm run build` passed; `git diff --check` passed. The new tests verify maximum-magnitude values are accepted, a generated finite target above `1e6` returns `DATASET_VALUE_OUT_OF_RANGE`, production HTTPS POST preflight/CORS behavior, exact-limit and chunked oversized request handling, malformed JSON envelopes, and finite success serialization. The backend suite continues to emit the existing Starlette/HTTPX deprecation warning.

## 2026-11-19 — Milestone 2 Linear Regression backend engine

- Added the first stateless `POST /api/v1/experiments/linear-regression` API using typed Pydantic request/response schemas, a thin route, a stateless orchestration service, and a FastAPI-independent NumPy engine.
- Contract example: `{"dataset":{"kind":"points","x":[-1,0,1],"y":[-1,1,3]},"parameters":{"initial_slope":0,"initial_intercept":0,"learning_rate":0.1,"steps":1},"seed":42}` returns `algorithm`, the effective `dataset`, `seed`, initial `parameters`, final `model`, `predictions.y_hat`, `metrics.mse`, and `training.steps_completed` plus `training.loss_history`. For that fixture it returns slope `4/15`, intercept `0.2`, MSE about `2.64296296296`, and history `[3.6666666667, 2.6429629630]`.
- Implemented inline and seed-driven synthetic datasets, float64 predictions and MSE, explicit simultaneous batch-gradient updates, exact `steps + 1` loss history, and parameter-based stateless continuation.
- Mathematical rule: `prediction = slope*x + intercept`, `MSE = mean((prediction-y)^2)`, `d_slope = 2*mean((prediction-y)*x)`, and `d_intercept = 2*mean(prediction-y)`. Both gradients are applied simultaneously from the pre-update parameters. Resubmit the effective dataset and returned slope/intercept to continue without server state.
- Module responsibilities: `api/routes/experiments.py` maps HTTP to schemas/service; `schemas/linear_regression.py` validates discriminated request variants and response sizes; `services/linear_regression.py` resolves the dataset and assembles output; `ml/linear_regression.py` contains NumPy data generation, training, predictions, metrics, and numeric checks; `middleware.py` enforces body size.
- Added finite checks for predictions, residuals, both gradients, candidate parameters, and losses. Structured validation, 1 MiB body-limit, and `NUMERICAL_DIVERGENCE` errors identify the first failing field and step without returning partial results. CORS now permits the experiment POST and its preflight.
- Added bounded `numpy>=1.26,<3` and `scikit-learn>=1.5,<2` dependencies. NumPy performs the transparent float64 vector math; scikit-learn's `LinearRegression` is a converged reference only (slope/intercept absolute tolerance `1e-8`), while gradient steps remain explicit NumPy calculations.
- Added comprehensive pytest coverage for fixture math, updates/losses, generated-data seeds, continuation, validation/work limits, oversize requests, divergence, finite large intermediates, API behavior, and sklearn comparison. Updated README with endpoint examples, math, limits, errors, and module responsibilities.
- Verification: backend pytest passed (30 tests); frontend `npm test` passed (6 tests); frontend `npm run build` passed. FastAPI TestClient smoke requests returned 200 for the documented happy path (slope `0.26666666666666666`, intercept `0.2`, MSE `2.642962962962963`), 422 `VALIDATION_ERROR` for invalid steps, and 422 `NUMERICAL_DIVERGENCE` at step 12 / `loss` for finite divergence inputs. The frontend remains unchanged and connected only to health until its later visualization milestone. Backend pytest reports one Starlette deprecation warning about using HTTPX with `TestClient`.
- No performance benchmark has been run; PRD performance targets remain unverified. Known limitation: one-feature batch gradient descent intentionally does not scale features or pre-reject unstable but finite learning-rate/data combinations; finite increasing loss is returned as requested. Dependency install was performed only in the ignored backend virtual environment.

## 2026-11-19 — Milestone 1 review fixes

- Made CORS origins environment-configurable through `CORS_ALLOWED_ORIGINS`; development falls back to the two Vite localhost origins, while production requires explicit origins and rejects wildcard or malformed origins. Enabled GET and OPTIONS handling; future POST routes must be added to the allow-method list and tested.
- Added `backend/.env.example`. The app reads process environment variables directly; local PowerShell commands and production platform configuration are documented in the README.
- Added pull-request and `main` push CI jobs for frontend tests/build and Python 3.11 backend tests. No secrets are used.
- Pinned the supported frontend runtime to Node.js 24.22.3 LTS in `.nvmrc` and constrained the frontend package engine to 24.22.x. A semver check confirmed this release satisfies the installed Vite, Vitest, jsdom, React plugin, and project engine ranges. Dependencies were not changed.
- Documented static build output, platform-provided FastAPI `PORT`, production API/CORS environment variables, and a deployment smoke-test checklist. No host was selected and no deployment was made.
- Verification: `npm test` passed (6 tests); `npm run build` passed; backend pytest passed (6 tests); CI YAML parsed and its triggers/working directories were asserted; `git diff --check` passed. Tests ran locally on Node 25.2.0 (outside the supported project range) and passed; CI selects the pinned Node LTS. Backend pytest reports Starlette's deprecation warning for HTTPX TestClient.

# 2026-11-19 — Milestone 1 scaffold

- Added the React/Vite/TypeScript frontend with Tailwind CSS v4's Vite plugin, a responsive landing page, typed health API client, retryable connection state, and Vitest/React Testing Library tests. Plotly is not installed.
- Added a FastAPI app with an isolated `/api/v1/health` router, Pydantic response model, Vite-local CORS allowlist, bounded dependency ranges, and pytest route/CORS tests.
- Added `frontend/.env.example` for the public `VITE_API_BASE_URL` setting and PowerShell setup/run/test instructions in the README.
- Verification: `npm test` passed (2 files, 6 tests); `npm run build` passed (TypeScript check plus Vite production build); `backend/.venv/Scripts/python.exe -m pytest` passed (2 tests); live Uvicorn `GET /api/v1/health` returned `{"status":"ok"}`. The pytest CORS preflight assertion also passed.
- Environment notes: Node 25.2.0 emitted EBADENGINE warnings because the current Vitest/jsdom package ranges target supported LTS releases; tests and build nevertheless passed. Backend pytest emitted a Starlette deprecation warning about using HTTPX with TestClient; tests passed. Use a supported Node LTS release for routine development.

## 2026-11-19 — Repository planning baseline

- Inspected the initialized repository on `main`; it had no application source files and contained an existing `.vscode/extensions.json` user file.
- Recorded product scope, stack, quality requirements, and module boundaries in the planning documents.
- Added ignore rules for dependencies, generated build/test artifacts, environment files, and `.vscode/`.
- No application scaffold, tests, commit, or push were created as part of this baseline.

## 2026-11-19 — Full-stack architecture update

- Documented a `frontend/` and `backend/` monorepo with a stateless FastAPI service and separate Python ML engine.
- Specified REST/Pydantic contracts, seeds, provisional input/work limits, and frontend, backend, and integration testing plans.
- Kept database, authentication, Redis, Celery, and LLM integration out of MVP. No code was scaffolded.

## 2026-11-19 — Final pre-implementation documentation

- Finalized the frontend plan as React, Vite, TypeScript, Tailwind CSS, Plotly.js, and static frontend hosting plus a separate FastAPI deployment.
- Corrected data dimensions: Linear Regression V1 is one feature `x` and continuous target `y`; Logistic Regression and KNN use two features and labels; K-Means uses two features without labels.
- Defined Linear Regression V1 request/response examples, finite/range/size validation, exact batch gradient descent update and loss-history semantics, and seeded synthetic generation.
- Added measurable performance targets and a benchmark evidence policy; targets remain unverified until measured on a recorded reference environment.
- Updated `.gitignore` for frontend outputs, Python caches/virtual environments, secrets, and editor-local settings. No application was scaffolded and no commit or push was made.
