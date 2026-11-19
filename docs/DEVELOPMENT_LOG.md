# Development Log

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
