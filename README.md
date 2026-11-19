# ML Playground

ML Playground is an interactive platform for beginner-to-intermediate learners to explore foundational machine learning through data, controls, visualizations, and explanations.

## MVP

Algorithms ship in this order: Linear Regression, Logistic Regression, K-Nearest Neighbors (KNN), then K-Means. Learners adjust datasets and parameters, inspect mathematically grounded results, and follow learning-step calculations where supported. Experiments are reproducible with explicit random seeds.

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

## Project status

This repository contains planning documents only; application code has not been scaffolded. Setup, local development, and deployment commands will be added when implementation begins.
