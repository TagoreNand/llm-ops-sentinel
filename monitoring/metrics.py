"""
Prometheus Custom Metrics

All LLM-specific metrics exposed at /metrics for Prometheus scraping.
Grafana dashboards consume these for real-time observability.
"""
from prometheus_client import Counter, Histogram, Gauge

# ── Counters ──────────────────────────────────────────────────────────────────

llm_calls_total = Counter(
    "llm_calls_total",
    "Total number of LLM API calls",
    ["model"],
)

llm_cost_dollars = Counter(
    "llm_cost_dollars_total",
    "Cumulative LLM API spend in USD",
    ["model"],
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total tokens used",
    ["model", "type"],  # type: input | output
)

drift_alerts_total = Counter(
    "drift_alerts_total",
    "Number of drift alerts triggered",
)

rollback_events_total = Counter(
    "rollback_events_total",
    "Number of automatic prompt rollbacks",
    ["prompt_name"],
)

eval_tasks_total = Counter(
    "eval_tasks_total",
    "Total evaluation tasks completed",
    ["status"],  # status: success | failure
)

# ── Histograms ────────────────────────────────────────────────────────────────

llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM API call latency in seconds",
    ["model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

eval_score_histogram = Histogram(
    "eval_overall_score",
    "Distribution of LLM evaluation scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Gauges ────────────────────────────────────────────────────────────────────

prompt_avg_score = Gauge(
    "prompt_avg_score",
    "Current average evaluation score for a prompt version",
    ["prompt_name", "version"],
)

drift_score_current = Gauge(
    "drift_score_current",
    "Most recent semantic drift score (JS divergence)",
)

active_canary_versions = Gauge(
    "active_canary_versions",
    "Number of prompt versions currently in canary rollout",
)


def setup_metrics():
    """Called on app startup to initialise default label sets."""
    for model in ["gpt-4o", "gpt-3.5-turbo", "claude-haiku-3-5", "claude-sonnet-4-6"]:
        llm_calls_total.labels(model=model)
        llm_latency_seconds.labels(model=model)
        llm_cost_dollars.labels(model=model)
        llm_tokens_total.labels(model=model, type="input")
        llm_tokens_total.labels(model=model, type="output")
