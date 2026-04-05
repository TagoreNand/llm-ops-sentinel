# LLM Ops Sentinel

> Production-grade LLM observability, evaluation, and self-healing prompt versioning pipeline.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![Docker](https://img.shields.io/badge/Docker-Compose-blue) ![MLflow](https://img.shields.io/badge/MLflow-2.13-orange)

## What it does

LLM Ops Sentinel sits as a proxy between your application and LLM APIs. Every call is:
1. **Intercepted & logged** — prompt hash, tokens, cost, latency persisted to Postgres
2. **Evaluated asynchronously** — GPT-4o judge scores faithfulness, relevance, and toxicity via Celery workers
3. **Monitored for drift** — nightly semantic embedding comparison alerts when output distributions shift
4. **Cost-optimised** — a complexity classifier routes simple queries to cheaper models (~40% cost reduction)
5. **Version-controlled** — prompts have Git-style canary rollouts with automatic rollback on score regression

## Architecture

```
Client App
    │
    ▼
┌─────────────────┐        ┌──────────────┐
│  FastAPI Proxy  │───────▶│  LLM APIs    │
│  (middleware)   │        │  (OAI/ANT)   │
└────────┬────────┘        └──────────────┘
         │
    ┌────┴─────────────────────────┐
    │                              │
    ▼                              ▼
┌──────────┐              ┌───────────────┐
│ Postgres │              │ Redis / Celery│
│ (logs)   │              │ (eval queue)  │
└──────────┘              └───────┬───────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ LLM Judge│ │  Drift   │ │  MLflow  │
              │ Evaluator│ │ Detector │ │ Tracking │
              └──────────┘ └──────────┘ └──────────┘
                                              │
                                    ┌─────────▼────────┐
                                    │Prometheus+Grafana │
                                    └──────────────────┘
```

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/yourname/llm-ops-sentinel
cd llm-ops-sentinel
cp .env.example .env
# Edit .env with your API keys

# 2. Start the full stack
docker-compose up --build

# 3. Test the proxy
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain transformers in one sentence", "model": "auto"}'

# 4. View dashboards
open http://localhost:3000   # Grafana  (admin/admin)
open http://localhost:5001   # MLflow
open http://localhost:8000/docs  # API docs
```

## Project Structure

```
llm-ops-sentinel/
├── app/
│   ├── main.py               # FastAPI app + proxy middleware
│   ├── config.py             # Settings (pydantic-settings)
│   ├── database.py           # SQLAlchemy async engine
│   ├── api/
│   │   ├── proxy.py          # LLM call interception & routing
│   │   └── prompts.py        # Prompt version registry endpoints
│   ├── core/
│   │   ├── router.py         # Cost-aware model router
│   │   ├── cost.py           # Token cost calculator
│   │   └── hasher.py         # Prompt hashing utilities
│   └── evaluators/
│       └── judge.py          # LLM-as-judge async evaluator
├── workers/
│   ├── celery_app.py         # Celery configuration
│   └── tasks.py              # Evaluation + alerting tasks
├── drift/
│   ├── embedder.py           # Sentence-transformer embeddings
│   └── detector.py           # UMAP + HDBSCAN drift detection
├── monitoring/
│   ├── metrics.py            # Prometheus custom metrics
│   └── alerts.py             # Slack / PagerDuty webhooks
├── tests/
│   ├── conftest.py
│   ├── test_proxy.py
│   ├── test_router.py
│   ├── test_evaluator.py
│   └── test_drift.py
├── scripts/
│   └── seed_golden_set.py    # Seed initial evaluation dataset
├── docker-compose.yml
├── Dockerfile
├── prometheus.yml
└── .github/workflows/ci.yml
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | required |
| `ANTHROPIC_API_KEY` | Anthropic API key | required |
| `DATABASE_URL` | Postgres connection string | see .env.example |
| `REDIS_URL` | Redis connection string | redis://redis:6379 |
| `SLACK_WEBHOOK_URL` | Drift alert webhook | optional |
| `MLFLOW_TRACKING_URI` | MLflow server URI | http://mlflow:5000 |
| `DRIFT_THRESHOLD` | Cosine similarity alert threshold | 0.15 |
| `ROLLBACK_SCORE_THRESHOLD` | Min eval score before rollback | 0.65 |

## Running Tests

```bash
# Unit + integration tests
docker-compose run --rm app pytest tests/ -v --cov=app --cov-report=term-missing

# Just unit tests (no external calls)
pytest tests/ -v -m "not integration"
```

## Key Design Decisions

- **Async everywhere** — FastAPI + asyncpg + httpx for non-blocking I/O
- **Evaluation is async** — never adds latency to the critical path (Celery workers)
- **Redis as prompt registry** — sub-millisecond reads for prompt version lookups
- **Canary rollouts** — new prompt versions serve 10% traffic until eval scores stabilise
- **MLflow for experiment tracking** — compare prompt versions like ML experiments
