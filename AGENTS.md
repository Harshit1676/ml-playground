# Contributor and agent guidance

## Product boundaries

- Build for beginner-to-intermediate learners. The public launch target is Linear Regression, Logistic Regression, then K-Means; plan KNN for a later release.
- Keep MVP experiments stateless and reproducible with explicit seeds.
- Do not add authentication, a database, Redis, Celery, or AI chatbot/LLM integration.
- Charts and explanations must describe results computed from the same request and dataset.

## Architecture

- Use the `frontend/` and `backend/` monorepo structure.
- Frontend: React, Vite, TypeScript, Tailwind CSS, Plotly.js. It owns controls, user interaction, visualization, and explanation presentation.
- Backend: Python 3.11+, FastAPI, Pydantic, NumPy, scikit-learn. A separate Python ML engine owns dataset generation, training, predictions, metrics, and learning-step calculations. Never put ML computation in React.
- Separate frontend UI state, API client/types, visualization adapters, and explanation presentation.
- Separate backend API routes, request/response schemas, services, and ML algorithms. Keep algorithm code independent from HTTP concerns.
- Validate schemas, finite values, feature dimensions, parameter ranges, seeds, payload sizes, and work limits at the API boundary. Return typed errors.
- Use scikit-learn as a reference where appropriate. Document intentional differences, especially when transparent step calculations are needed for teaching.
- Dataset dimensions: Linear Regression V1 has one input feature `x` and one continuous target `y`; the current Logistic Regression lab has one feature `x` and a binary label `y`; K-Means has two input features and no labels. Future two-feature Logistic Regression and KNN experiences use two input features plus labels.

## Testing and performance

- Frontend tests use Vitest and React Testing Library for state, controls, API client behavior, visualization transforms, and explanations.
- Backend tests use pytest for schemas, routes, services, algorithms, invariants, edge cases, and determinism.
- Integration tests verify frontend/API contracts and representative seeded experiments.
- Check chart axes, decision boundaries, probabilities, assignments, and metrics against backend results. Preserve keyboard access and do not encode meaning by color alone.
- Performance targets in the PRD are goals, not claims. Benchmark the defined workloads on a recorded reference environment before reporting that a target is achieved.

## Mandatory implementation and learning workflow

For each implementation task, follow this sequence:

1. **Implement** the agreed change while preserving unrelated user work.
2. **Test** the change and report the exact checks and outcomes; do not claim unrun tests passed.
3. **Summarize** files and behavior changed.
4. **Explain** the design choices and relevant tradeoffs in clear terms.
5. **Provide interview questions** about the concepts and decisions in the change so the learner can check their understanding.
6. **Request explicit approval** after presenting the reviewable result.
7. **Commit and/or push only after approval.** If approval is not given, stop before either action.

If the task explicitly excludes commits or pushes, honor that instruction even after approval is requested. Never commit or push implicitly.

## Repository workflow

- Inspect existing files before editing; do not overwrite unrelated work.
- Keep dependency folders, build output, Python caches/virtual environments, secrets, and editor-local settings out of Git as configured in `.gitignore`.
- Keep docs aligned with implementation and record meaningful milestones in `docs/DEVELOPMENT_LOG.md`.
