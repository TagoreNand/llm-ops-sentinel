# LLM Ops Sentinel

> Production-oriented LLM observability, evaluation, drift monitoring, and prompt operations platform for teams running multiple LLM-powered features in production.

LLM Ops Sentinel is a FastAPI-based control plane that sits between an application and external LLM providers. It captures every model call, routes requests to a cost-appropriate model, stores operational telemetry, evaluates outputs asynchronously, tracks experiment results in MLflow, exposes Prometheus metrics, and runs scheduled checks for semantic drift and canary prompt quality.

---

## Why this project exists

Shipping LLM applications is no longer just about generating a response. In production, teams need to answer questions like:

- Which model handled the request and why?
- How much did the call cost?
- How long did it take?
- Was the output relevant, faithful, and safe?
- Are responses drifting over time?
- Can we test prompt changes gradually before promoting them?

LLM Ops Sentinel turns those concerns into a single operational workflow.

---

## What the platform does

### 1) Acts as an LLM gateway
The service exposes a `/v1/chat` endpoint that receives prompts, optionally accepts a system message, and either honors a caller-specified model or automatically routes the request to a cheaper or more capable model based on prompt complexity.

### 2) Logs every request for auditability
Each call is persisted with a prompt hash, full prompt text, response text, selected model, token counts, latency, cost, and metadata so teams can inspect model behavior historically.

### 3) Evaluates outputs asynchronously
After the response is returned to the caller, a Celery worker evaluates the output using an LLM-as-a-judge workflow. Scores are stored in Postgres and also logged to MLflow for experiment tracking.

### 4) Watches for drift in response behavior
A scheduled worker periodically embeds recent responses, reduces them into a lower-dimensional semantic space, compares their distribution against a stored baseline, and raises alerts when distribution shift crosses a threshold.

### 5) Supports prompt operations
The project includes a prompt registry API that can create prompt versions, expose active and canary variants, resolve versions based on traffic split, and promote or roll back versions.

### 6) Exposes operational telemetry
The platform publishes Prometheus metrics for request volume, latency, token consumption, spend, evaluation score distributions, drift, and canary state, making it easy to wire dashboards into Grafana.

---

## End-to-end pipeline

```mermaid
flowchart TD
    A[Client / Product Application] --> B[FastAPI Proxy - /v1/chat]
    B --> C[Complexity Scoring + Cost-Aware Router]
    C --> D[OpenAI or Anthropic Model]
    D --> E[Response Returned to Client]

    B --> F[Postgres - LLMCall logs]
    B --> G[Prometheus Metrics]
    B --> H[Celery Queue]

    H --> I[LLM-as-Judge Evaluator]
    I --> J[Postgres - EvaluationResult]
    I --> K[MLflow Experiment Tracking]

    L[Celery Beat Scheduler] --> M[Nightly Drift Detection]
    M --> N[Sentence Embeddings]
    N --> O[UMAP + HDBSCAN]
    O --> P[Jensen-Shannon Divergence]
    P --> Q[Postgres - DriftEvent]
    P --> R[Slack / PagerDuty Alerts]

    S[Prompt Registry API] --> T[Create / Resolve / Promote / Rollback]
    T --> U[Redis Prompt Cache]
    T --> V[Postgres PromptVersion Audit Trail]
```

---

## Request lifecycle

### Step 1: Application sends a prompt
A product service sends a request to `POST /v1/chat` with a prompt, optional system instructions, generation settings, and either a fixed model or `auto` routing.

### Step 2: The router selects a model
The router scores the prompt using lightweight heuristics such as prompt length, question depth, and complexity keywords. Based on that score, it selects one of several configured model tiers.

### Step 3: The selected provider is called
The proxy dispatches the request to the appropriate provider SDK:
- OpenAI-compatible models are sent through the OpenAI client.
- Claude models are sent through the Anthropic client.

### Step 4: Operational metadata is computed
Once the provider responds, Sentinel calculates:
- input tokens
- output tokens
- latency in milliseconds
- estimated cost in USD
- prompt hash for deduplication and traceability

### Step 5: The request is persisted
The completed call is written into Postgres as an `LLMCall` record for audit history and downstream analytics.

### Step 6: Metrics are emitted
The proxy increments Prometheus counters and histograms for usage, latency, spend, and token volume.

### Step 7: Evaluation is queued asynchronously
A Celery task is enqueued so response quality scoring does not block the user-facing request path.

### Step 8: Quality scoring runs in the background
The evaluation worker uses a GPT-4o-based judge prompt to score:
- faithfulness
- relevance
- toxicity
- overall composite score

If the judge call fails, the evaluator falls back to a heuristic scoring strategy instead of failing the pipeline.

### Step 9: Experiment tracking is recorded
The evaluation worker stores the scores in Postgres and logs the same run to MLflow so teams can inspect evaluation histories and compare models or prompt variants over time.

### Step 10: Scheduled governance jobs run later
Separately from real-time serving, Celery Beat triggers recurring jobs for:
- semantic drift detection
- canary prompt score checks
- alerting hooks

---

## Prompt operations pipeline

The repository contains a dedicated prompt control plane under `/prompts`.

### Supported workflow
1. Create a new prompt version.
2. Mark it as a canary or make it immediately active.
3. Cache active/canary prompt data in Redis for fast lookup.
4. Resolve the prompt version based on configured traffic split.
5. Promote a canary to active.
6. Roll back to a known version when needed.

### Why this matters
This gives the project a GitOps-style prompt management layer, which is useful when prompt changes are treated like deployable assets rather than ad hoc strings in application code.

---

## Drift detection pipeline

The drift subsystem is built for semantic monitoring instead of simple token-level logging.

### How it works
1. Fetch the latest response corpus from Postgres.
2. Convert responses into sentence embeddings using `all-MiniLM-L6-v2`.
3. Reduce embeddings with UMAP.
4. Cluster the projected space with HDBSCAN.
5. Convert the semantic space into a binned distribution.
6. Compare the current distribution to the baseline using Jensen-Shannon divergence.
7. Persist the event and send alerts when the drift score crosses the configured threshold.

### Why the design is useful
This approach is stronger than raw average-length or token-frequency checks because it can catch shifts in response style, topic clusters, or semantic modes even when basic usage metrics still look normal.

---

## Tech stack

### Serving layer
- FastAPI
- Uvicorn
- Pydantic / pydantic-settings

### Data and state
- PostgreSQL
- Redis
- SQLAlchemy async engine

### Background processing
- Celery
- Celery Beat

### LLM providers and evaluation
- OpenAI SDK
- Anthropic SDK
- GPT-4o as judge
- sentence-transformers

### Drift analytics
- all-MiniLM-L6-v2 embeddings
- UMAP
- HDBSCAN
- NumPy / scikit-learn ecosystem

### Observability and MLOps
- Prometheus
- Grafana
- MLflow
- Slack / PagerDuty alert hooks

### Packaging and local infrastructure
- Docker
- Docker Compose
- GitHub Actions CI

---

## Repository structure

```text
llm-ops-sentinel/
├── app/
│   ├── main.py                 # FastAPI app startup, health, metrics mount, router registration
│   ├── config.py               # Environment-driven settings and model cost map
│   ├── database.py             # Async engine and ORM models
│   ├── api/
│   │   ├── proxy.py            # /v1/chat request interception, provider dispatch, logging, metrics
│   │   └── prompts.py          # Prompt version registry and canary/rollback endpoints
│   ├── core/
│   │   ├── router.py           # Complexity scoring and cost-aware model routing
│   │   ├── cost.py             # Cost calculation helpers
│   │   └── hasher.py           # Prompt hashing utilities
│   └── evaluators/
│       └── judge.py            # GPT-4o judge evaluator with heuristic fallback
├── workers/
│   ├── celery_app.py           # Celery config, queues, and schedules
│   └── tasks.py                # Async evaluation, drift detection, rollback checks
├── drift/
│   ├── embedder.py             # Sentence-transformer embedding generation
│   └── detector.py             # UMAP + HDBSCAN + JSD drift detection
├── monitoring/
│   ├── metrics.py              # Prometheus counters, histograms, gauges
│   └── alerts.py               # Slack and PagerDuty notifications
├── scripts/
│   └── seed_golden_set.py      # Seeds a small golden evaluation dataset
├── tests/                      # Unit and integration tests
├── docker-compose.yml          # Full local stack orchestration
├── Dockerfile                  # App image build
├── prometheus.yml              # Prometheus scrape config
├── requirements.txt            # Python dependencies
└── .github/workflows/ci.yml    # CI test workflow
```

---

## Data model

The database schema is organized around four main entities:

### `LLMCall`
Stores the raw operational event for each model request.

### `EvaluationResult`
Stores post-hoc quality scores for a completed LLM call.

### `PromptVersion`
Stores prompt templates, canary status, traffic split, and aggregate score metadata.

### `DriftEvent`
Stores semantic drift checks and alert outcomes.

This schema makes the project easy to extend into dashboards, scorecards, anomaly analysis, and prompt release governance.

---

## Local development

### 1. Clone and configure
```bash
git clone https://github.com/TagoreNand/llm-ops-sentinel.git
cd llm-ops-sentinel
cp .env.example .env
```

### 2. Fill environment variables
At minimum, configure:
```env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
DATABASE_URL=postgresql+asyncpg://sentinel:sentinel@postgres:5432/sentinel_db
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2
MLFLOW_TRACKING_URI=http://mlflow:5000
DRIFT_THRESHOLD=0.15
ROLLBACK_SCORE_THRESHOLD=0.65
CANARY_TRAFFIC_PERCENT=10
APP_ENV=development
```

### 3. Start the full stack
```bash
docker-compose up --build
```

### 4. Available local services
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Metrics endpoint: `http://localhost:8000/metrics`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- MLflow UI: `http://localhost:5001`

### 5. Send a test request
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "Explain transformers in one sentence",
        "model": "auto"
      }'
```

---

## API overview

### Chat proxy
`POST /v1/chat`

Primary runtime endpoint for LLM inference through Sentinel.

### Prompt registry
- `POST /prompts/versions`
- `GET /prompts/versions/{name}`
- `GET /prompts/resolve/{name}`
- `POST /prompts/versions/{name}/{version}/promote`
- `POST /prompts/versions/{name}/{version}/rollback`

### Platform endpoints
- `GET /health`
- `GET /metrics`

---

## Observability and dashboards

Sentinel is designed so that both engineers and ML teams can inspect the same system from different angles.

### For platform engineers
Use Prometheus and Grafana to track:
- throughput by model
- latency percentiles
- cumulative spend
- token usage trends
- drift alert frequency

### For ML / product teams
Use MLflow to inspect:
- evaluation score histories
- model-to-model comparisons
- canary experiment behavior
- regressions over time

---

## Testing and CI

The repository includes a GitHub Actions CI workflow that installs system dependencies, installs Python dependencies from `requirements.txt`, and runs the test suite with `pytest` on pushes and pull requests to `main`.

---

## What is implemented today vs. what is ready to extend

To keep the project description accurate, it helps to separate the current implementation from the next logical integrations.

### Implemented now
- Real-time proxying of OpenAI and Anthropic requests
- Cost-aware routing for `auto` requests
- Postgres logging of model calls
- Asynchronous response evaluation
- MLflow run tracking
- Prometheus instrumentation
- Scheduled drift detection
- Prompt version registry APIs
- Manual promote / rollback endpoints
- Slack and PagerDuty alert hooks
- Docker Compose stack for local operation

### Ready to wire in next
- Automatic prompt resolution inside `/v1/chat`
- Closed-loop rollback execution directly from the rollback worker
- Automatic prompt call counting for canary health decisions
- Golden-set comparison tied directly into online eval gates
- Multi-tenant auth and per-application isolation
- Human review queue for failed evaluations or drift events

---

## How is this project well-rounded

This project demonstrates more than LLM API usage. It shows understanding of:

- production inference orchestration
- asynchronous evaluation pipelines
- observability-first system design
- prompt release governance
- cost optimization for model selection
- semantic monitoring and drift detection
- Dockerized local platform engineering
- MLOps-style experiment tracking


---

## Suggested future enhancements

- Add JWT or API key authentication for multi-user deployments.
- Persist drift baselines in Postgres or object storage rather than `/tmp`.
- Add a reviewer UI for failed or low-score generations.
- Increment prompt-version usage counters directly from the inference path.
- Connect rollback checks to the prompt registry so rollback becomes truly automatic.
- Add tracing with OpenTelemetry.
- Add batch/offline evaluation jobs for benchmark datasets.
- Add support for RAG-specific metrics such as groundedness against retrieved context.

---

## License

MIT License
