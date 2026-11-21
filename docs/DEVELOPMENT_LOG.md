# Development Log

## Milestone 6B phase 1 ? release guards and binary label validation

- All four API clients share development URL resolution. Vite validates the effective release URL before bundling. CI builds against the reserved HTTPS test destination and has no artifact publication step.
- Source inspection identified a float conversion before binary label membership validation. A sufficiently large integer could raise OverflowError there. No pre-fix HTTP 500 was observed.
- At the schema boundary the corrected validator rejects integers outside float range with a Pydantic validation error. Through the FastAPI HTTP test client, valid JSON under 1 KiB containing a 400-digit numeric label returned 422 with the standard VALIDATION_ERROR envelope and a field-specific reason. The client propagates unhandled application exceptions, and none escaped. Numeric zero and one were accepted. Boolean, string and nonfinite labels were rejected.
- Node 24.22.3 passed 135 frontend tests. TypeScript and the configured Vite release build passed. Eight actual Vite invocations rejected invalid or missing destinations. The resulting successful bundle contained the verification URL and excluded the localhost fallback. Python 3.11.9 passed 130 backend tests.
- The [admission-control proposal](ADMISSION_CONTROL_PROPOSAL.md) records candidate process, gateway and hosting limits and staging acceptance tests for approval.

## 2026-11-20 — Learning lab readability and accessibility

- Moved primary plots ahead of model summaries and put Run above the controls. Mobile settings collapse on submission and reopen for field errors. Matched draft and submitted request payloads to flag stale results, including edits made while a request is pending.
- Linear Regression gained three reproducible dataset recipes, pointer and keyboard observation selection, and a signed residual based on the returned prediction. Logistic Regression shows predicted-class changes relative to the submitted threshold. K-Means compares adjacent backend snapshots for centroid trails, travel distances, and the count of reassigned observations.
- Compared all three default runs in local Chrome with FastAPI at 1440 × 900, 1280 × 800, and 390 × 844. At 1280 px wide the plot tops moved from 814 to 360 px (Linear), 1,389 to 360 px (Logistic), and 654 to 381 px (K-Means). At 390 px wide they moved from 2,404 to 511 px, 3,273 to 532 px, and 1,655 to 522 px respectively. Saved 18 full-resolution screenshots and a comparison gallery in the local review output.
- Checked input edits and reversions, keyboard focus, chart selection, threshold/iteration controls, loading, validation, and offline retry. The nine default views had no horizontal page overflow or JavaScript exceptions. Exploration produced zero extra training POSTs. Sampled text contrast reached at least 4.82:1 across the nine views, with plot text at 9.46:1 or higher. Helper text increased from 9 to 13 px and field labels from 10 to 14 px. Physical devices and screen readers remain untested.
- Node 24.22.3 passed all 111 frontend tests across 14 files and the TypeScript/Vite build. Backend pytest passed all 117 tests. Vite still warns about the deferred Plotly chunk (1,160.65 kB minified, 384.84 kB gzip). Starlette still warns about the TestClient HTTPX dependency.

## Milestone 5B — interactive K-Means lab

- Added the third algorithm link, inline and synthetic forms, runtime response checks and a manual snapshot explorer. Every displayed centroid, assignment and inertia comes from the accepted backend run. Both charts follow the same selected iteration. A text table exposes empty clusters, overlapping centroids and individual assignments.
- Node 24.22.3: all 107 frontend tests across 14 files passed, and the TypeScript/Vite production build passed. Backend pytest passed all 117 tests with the existing Starlette TestClient deprecation. The lazy Plotly asset measured 1,160.44 kB minified (384.74 kB gzip), triggering Vite's size warning.
- Headless Chrome used local FastAPI and Vite at desktop 1280 × 900 and emulated mobile 390 × 844. Checked the direct link, all three algorithm links, history traversal, retained form and step, default synthetic run, initialization-seed independence, inline runs, varying k, empty/coincident centers, capped results, invalid input, reset and offline/retry. Compared plotted assignments, centroids and selected inertia against HTTP responses. Both revealed mobile plots matched their 340 px containers. Checked for horizontal overflow and JavaScript exceptions/console errors.
- For the documented seed-3 maximum-workload fixture (1,000 points, 16 clusters, 101 snapshots), five desktop slider changes to steps 0, 25, 50, 75 and 100 took 264.4, 259.1, 274.0, 225.4 and 226.6 ms respectively through chart update plus two animation frames. No additional POSTs occurred. These local development measurements are exploratory. Physical mobile devices, other browser engines and production latency remain unmeasured.

## Milestone 5A contract corrections

- Standardized the route on `POST /api/v1/experiments/k-means`. OpenAPI now advertises request and response dataset variants keyed by `kind`.
- Added an unlabeled square-cluster generator. Each request uses its own generation RNG and the existing centroid initialization RNG. The synthetic response includes the generator recipe and effective points. Replaying those points inline reproduced every training snapshot.
- Verified 117 backend tests and 84 frontend tests across 12 files. The frontend production build passed. Both complete README request/response pairs matched FastAPI TestClient responses. Checks used Node 25.2.0 locally. The existing Starlette TestClient deprecation and deferred Plotly chunk-size warnings remain.

## Milestone 5A audit — iteration playback

- Added bounded, numbered snapshots with centroid coordinates, assignments and inertia. Fixed a rounding defect where averaging three identical decimal coordinates moved the center and produced positive inertia.
- Tested a mixed duplicate-coordinate dataset with five distinct locations and four sampled rows. One empty center later gained members and another stayed empty through convergence. Independent distance, mean, cluster count and inertia calculations check each state, including a run capped before convergence.
- Measured uncompressed FastAPI response bodies with 1,000 points, 16 clusters and seed 3. With `t = arange(1000) / 999`, points `(999999.123456789*t, -38231.2345678901*t)` reached all 100 updates. The 101-snapshot response was 356,749 bytes. With a cap of one update the same input returned 49,582 bytes. Both measurements include the dataset and final summary.
- Full verification passed with 95 backend tests, 84 frontend tests and a TypeScript/Vite production build. Local runtimes were Python 3.11.9 and Node 25.2.0. The existing Starlette TestClient deprecation and Vite Plotly chunk-size warning remain.

## Milestone 5A — 2D K-Means backend

- Added the stateless K-Means route, Pydantic contracts, service and float64 Lloyd engine. The engine returns initial and final centers, nearest-center assignments and inertia at initialization and after each update.
- The scikit-learn reference test supplies our initial centers, uses one Lloyd run and sets `tol=0`. This removes differences from scikit-learn's default k-means++ initialization and tolerance for center movement. Our convergence check uses unchanged assignments. Our engine keeps empty centers in place, whereas scikit-learn can relocate empty clusters using distant observations. The reference fixture avoids empty clusters. See the [scikit-learn KMeans reference](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
- From `backend/`, `.venv/Scripts/python.exe -m pytest -q` passed all 90 tests, including 24 K-Means cases. The existing Starlette TestClient deprecation warning remains. Tests cover a hand-calculated 2D update, repeatability, ties, duplicates, empty clusters, iteration limits, very small and large coordinates, structured validation errors and the scikit-learn comparison.

## 2026-11-20 - Milestone 4B pre-commit audit

- Fixed the Logistic true-intercept label targeting a hidden Linear input. Algorithm links now preserve other URL parameters, respect modified clicks and skip duplicate history entries. A local favicon resolves the 404 seen on first load.
- The response validator now accepts a generated feature rounded to `x_max`, rejects a changed zero-step model and checks final BCE against the returned model. A dropped connection while reading JSON offers retry, and an abort keeps its cancellation semantics.
- The chart now samples steep sigmoid transitions and deduplicates coordinates at adjacent floating-point values. Threshold readouts and observation probabilities retain the precision used to assign classes. The wrapped legend sits above the data at mobile width.
- The four-observation regression test checks predictions, confusion counts, all four classification metrics, the threshold guide, fixed model outputs and a single training POST. Other regressions cover threshold endpoints and equality, history traversal, label targets, invalid responses and chart coordinates.
- An isolated headless Chrome session exercised Vite and FastAPI on loopback. Both deep links, fallback for an unknown algorithm, keyboard focus, the accessibility tree, preserved results, Back/Forward, invalid input, Reset and retry after a simulated connection failure passed. On the seed-42 run, moving the threshold from 0.5 to 0.7 changed precision from 84.8% to 89.7% and recall from 83.0% to 74.5%, with one Logistic training POST in total.
- At a 390 × 844 emulated viewport, both revealed charts matched their containers and the page had no horizontal overflow. Chrome reported no JavaScript exceptions. Deliberately requesting generated single-class data produced the expected HTTP 422. Physical devices and other browser engines were outside this check.
- Final checks passed on Node.js 25.2.0 with 84 frontend tests across 12 files, a TypeScript/Vite production build, 66 backend tests and `git diff --check`. The deferred Plotly asset is 1,160.44 kB (384.74 kB gzip) and triggers Vite's chunk-size warning. Pytest reported the existing Starlette/HTTPX TestClient deprecation. The pinned Node.js 24.22.3 runtime still needs verification for this revision.

## 2026-11-19 - Milestone 4B Logistic Regression learning experience

- Added query-string algorithm navigation for Linear and Logistic Regression (`?algorithm=logistic-regression`), with keyboard-accessible links and active-page state. Both labs remain mounted during navigation to preserve their form/results; KNN and K-Means are not exposed as unfinished choices.
- Added a responsive one-feature Logistic Regression experiment form for seeded synthetic and inline binary datasets. It preserves inputs across dataset-mode changes, hides the seed for inline data while sending the required contract seed, validates request bounds, clears old results for invalid submissions, supports retry/reset, cancels older fetches, and ignores stale responses.
- Added an independent typed client for the existing M4A POST endpoint. Runtime checks cover response fields, echoed parameters and inline observations, dataset dimensions/ranges, both synthetic classes, probabilities derived from returned model parameters, threshold classes, confusion counts/metrics, finite nonnegative BCE/loss, step counts, and step-zero BCE derived from submitted initial parameters. No backend contract or source changed.
- Added response-derived probability/observation, threshold-guide, and BCE charts with the existing lazy-loaded Plotly Basic bundle. Threshold exploration derives classes, confusion matrix, accuracy, precision, recall, and F1 locally without retraining; coefficient/intercept, probabilities, BCE, and loss history remain fixed. The UI explains probabilities versus classes, confusion-matrix orientation/terms, zero-denominator metrics, and why accuracy is not sufficient by itself.
- Added request, API, chart-transform, threshold-metric, component, navigation, and stale-response tests, including threshold boundary and steps-zero cases. No dependencies were added or changed. The implementation remains one-feature only, has no coefficient/intercept training history or animation, and still depends on the running FastAPI service.
- Verification: frontend `npm test` passed (12 files, 73 tests); `npm run build` passed; backend `backend/.venv/Scripts/python.exe -m pytest -q` passed (66 tests); `git diff --check` passed. A live local HTTP check returned 200 for the Vite deep-link URL and the seeded Logistic Regression POST, with the expected dev CORS origin, both labels, 25 updates, and 26 loss values. The build emitted the existing Plotly deferred-chunk warning at 1,160.44 kB minified (384.74 kB gzip). Backend pytest emitted the existing Starlette warning: `Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.`
- Runtime and browser limits: checks ran with Node.js 25.2.0, outside the supported Node.js 24.22.3 version; no runtime requirement was changed. Browser UI verification was not completed because the current Chrome tab was on ChatGPT rather than the local app and a fresh window observation showed focus had moved to VS Code. No visual rendering, mobile layout, or browser-console result is claimed; the manual checklist is provided in the README. No performance benchmark or deployment was run.

## 2026-11-19 - Milestone 4A Logistic Regression backend

- Added a stateless `POST /api/v1/experiments/logistic-regression` endpoint with discriminated inline and seeded synthetic dataset schemas, bounded parameters, thresholded predictions, metrics, response consistency checks, and the shared structured error envelope.
- Implemented the univariate NumPy engine `z = coefficient*x + intercept` with a sign-stable sigmoid, logits-based BCE (`logaddexp(0,z) - y*z`), simultaneous full-batch updates, seeded uniform/Bernoulli synthetic data, and finiteness checks. Probabilistic synthetic output containing only one class returns `SINGLE_CLASS_DATASET`; inline single-class datasets remain valid and zero-denominator metrics are defined as `0.0`.
- Adversarial review found raw `np.mean` could overflow while summing large finite loss/gradient terms even when their mean was representable. Added a scaled finite-mean helper and regression coverage so the engine proceeds to the actual failing operation rather than reporting premature divergence. BCE now carries the current update step into its finite-value errors.
- This entry records the backend-only M4A stage; M4B now uses its univariate API for the learner UI. A two-feature classifier and decision-boundary visualization remain future extensions. Parameter history is not returned; a future learning animation must receive explicit parameter snapshots instead of inferring them from loss history. The default regularized scikit-learn classifier is not an exact reference for this unregularized explicit gradient descent.
- Backend tests: `backend/.venv/Scripts/python.exe -m pytest -q` passed, **66 passed**, with one existing Starlette deprecation warning: `Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.` Logistic Regression tests cover an independently calculated nontrivial one-step fixture, simultaneous updates, `steps: 0`, exact threshold behavior and post-training threshold independence, all confusion outcomes, seeded data range/repeatability, single-class outcomes, and finite aggregation under extreme values.
- Frontend regression checks: `npm test` passed (7 files, 44 tests); `npm run build` passed with Node.js 25.2.0. Vite emitted its existing warning for the 1,160.44 kB minified deferred Plotly chunk. No frontend application code changed.
- `git diff --check` passed. No live browser flow or performance benchmark was run; the Linear Regression learner UI remains unchanged.

## 2026-11-19 - Milestone 3 code review fixes

- Invalid form submissions now clear a prior successful result so stale metrics and charts are not shown alongside validation errors. The result seed badge is shown only for synthetic dataset results.
- The frontend response validator now checks that the backend echoes inline `x` and `y` observations from the submitted request, in addition to validating response dimensions and model predictions. This matches the current stateless backend service, which passes submitted inline arrays through unchanged.
- Verification: frontend `npm test` passed (7 files, 44 tests); `npm run build` passed with the existing Vite warning for the 1,160.44 kB minified deferred Plotly chunk; backend pytest passed (41 tests, one existing Starlette/HTTPX deprecation warning); `git diff --check` passed.

## 2026-11-19 - Milestone 3 interactive Linear Regression frontend

- Replaced the health-only landing experience with a responsive Linear Regression lab while retaining the backend health status and retry control. Added seeded synthetic and inline point datasets, dataset and training controls, field-level validation, and a reset action.
- Added a typed `POST /api/v1/experiments/linear-regression` client that sends the existing `dataset`, `parameters`, and `seed` contract, validates successful response shapes, shows structured backend errors, reports network/backend failures, supports retry, aborts superseded requests, and ignores stale responses. No backend API or ML computation was changed.
- Added result cards for the returned model slope/intercept, MSE, and completed steps. Plotly traces use only returned observations, final returned model parameters, and the returned loss history; no intermediate parameter values are inferred. Plotly is code-split and loaded when charts are displayed.
- Added `react-plotly.js` and `plotly.js-dist-min`. Added feature, API client, chart-transform, and app-shell tests for defaults, validation, payload construction, success/errors, loading, retry, reset, response-derived chart data, and stale requests.
- Verification: frontend `npm test` passed (6 files, 31 tests); `npm run build` passed (TypeScript check and Vite production build); backend pytest passed (41 tests); `git diff --check` passed. Backend pytest retains the existing Starlette deprecation warning: `Using httpx with starlette.testclient is deprecated; install httpx2 instead.`
- Build note: the initial app chunk is 244.02 kB minified (76.23 kB gzip). The deferred Plotly asset is 4,656.06 kB minified (1,423.60 kB gzip), so Vite reports its standard greater-than-500-kB chunk warning. It is deferred until results are shown, but remains a performance optimization opportunity; no performance benchmark was run.
- Runtime note: this workstation used Node.js v25.2.0, outside the repository's declared Node.js 24.22.x support range. Tests and build passed here; repeat CI/local verification on the pinned Node.js 24.22.3 LTS before treating the supported runtime as verified.
- Local preview: use the existing PowerShell commands in the README to run FastAPI and Vite in separate terminals. Automated frontend API tests use mocked fetch responses; a live deployed-service integration or browser screenshot was not performed.

## 2026-11-19 - Milestone 3 final corrections

- Set the noise slider step to `0.1`; the numeric field and range input both remain controlled by the same `noiseStd` form value. Inline datasets now hide the seed control while retaining the required top-level request `seed` (42 for inline requests); switching back to synthetic data restores the user's synthetic seed setting.
- Strengthened successful-response validation to reject non-finite/negative loss values, mismatched prediction counts, and predictions inconsistent with the returned slope/intercept using a relative tolerance of `1e-9` rather than exact float equality. The backend schema was not changed.
- Compared the original `plotly.js-dist-min@4.3.0` build (4,656.06 kB minified / 1,423.60 kB gzip for the deferred chunk) with the official `plotly.js-basic-dist-min@4.1.2` partial bundle (1,160.44 kB / 384.74 kB gzip). The basic bundle includes scatter marker and line modes plus the Plotly methods used by the React adapter; chart-transform and bundle-capability tests passed. Kept the full Plotly package only as a development type source. The smaller bundle still triggers Vite's >500 kB minified warning.
- GitHub Actions uses `actions/setup-node` with `.nvmrc`, which pins Node.js 24.22.3. This workstation has Node.js 25.2.0 only; tests/build here do not verify Node 24. CI will verify that runtime after these changes are pushed.
- Browser automation did not safely verify the local URL/target during this check. The README now contains a manual checklist for default runs, parameter changes, dataset switching, reset, validation, backend unavailability, mobile layout, and console errors. No browser flow is claimed as passed.
- Verification: frontend `npm test` passed (7 files, 41 tests); `npm run build` passed with a Vite warning for the 1,160.44 kB minified chart chunk; backend pytest passed (41 tests, one existing Starlette/HTTPX deprecation warning); `git diff --check` passed. These checks ran on Node.js 25.2.0, outside the declared 24.22.x range.

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
