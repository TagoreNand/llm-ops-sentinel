"""
Cost-aware model router.

Routes LLM calls to the cheapest model capable of handling the query.
Uses a lightweight complexity classifier based on query length, entity count,
and keyword signals — no external model call needed.
"""
import re
import random
from dataclasses import dataclass

import structlog

from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

# Ordered from cheapest to most capable
MODEL_TIERS = [
    "claude-haiku-3-5",
    "gpt-3.5-turbo",
    "claude-sonnet-4-6",
    "gpt-4o",
]

# Keywords that signal a complex query requiring frontier models
COMPLEXITY_SIGNALS = [
    r"\banalyze\b", r"\bcompare\b", r"\bexplain.*why\b", r"\bsummarize\b",
    r"\bdebug\b", r"\barchitect\b", r"\boptimize\b", r"\bcritique\b",
    r"\bcontrast\b", r"\binfer\b", r"\bhypothes", r"\bcode\b", r"\bimplement\b",
]

COMPLEXITY_PATTERN = re.compile("|".join(COMPLEXITY_SIGNALS), re.IGNORECASE)


@dataclass
class RoutingDecision:
    model: str
    complexity_score: float
    reason: str
    estimated_cost_per_1k: float


def score_complexity(prompt: str) -> float:
    """
    Returns a complexity score from 0.0 (trivial) to 1.0 (highly complex).
    Combines token count heuristic, complexity keyword signals, and question depth.
    """
    words = prompt.split()
    word_count = len(words)

    # Length signal (normalised to 0-0.4)
    length_score = min(word_count / 500, 1.0) * 0.4

    # Keyword signal (0 or 0.4)
    keyword_matches = len(COMPLEXITY_PATTERN.findall(prompt))
    keyword_score = min(keyword_matches / 3, 1.0) * 0.4

    # Question depth: multiple sentences / clauses (0-0.2)
    sentence_count = len(re.split(r"[.!?]", prompt))
    depth_score = min(sentence_count / 10, 1.0) * 0.2

    return round(length_score + keyword_score + depth_score, 3)


def select_model(complexity: float, force_model: str | None = None) -> tuple[str, str]:
    if force_model and force_model != "auto":
        return force_model, "forced by caller"

    if complexity < 0.25:
        return MODEL_TIERS[0], "simple query → cheapest model"
    elif complexity < 0.55:
        return MODEL_TIERS[1], "medium complexity → mid-tier model"
    elif complexity < 0.80:
        return MODEL_TIERS[2], "high complexity → capable model"
    else:
        return MODEL_TIERS[3], "very complex → frontier model"


def route(prompt: str, force_model: str | None = None) -> RoutingDecision:
    complexity = score_complexity(prompt)
    model, reason = select_model(complexity, force_model)

    costs = settings.model_costs.get(model, {"input": 0.001, "output": 0.002})
    estimated_cost = (costs["input"] + costs["output"]) / 2

    logger.info(
        "model_routed",
        model=model,
        complexity=complexity,
        reason=reason,
    )

    return RoutingDecision(
        model=model,
        complexity_score=complexity,
        reason=reason,
        estimated_cost_per_1k=estimated_cost,
    )
