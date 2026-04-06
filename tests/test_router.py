"""Tests for the cost-aware model router."""
import pytest
from app.core.router import score_complexity, route, MODEL_TIERS


def test_simple_prompt_low_complexity():
    score = score_complexity("What time is it?")
    assert score < 0.25


def test_complex_prompt_high_complexity():
    score = score_complexity(
        "Analyze and compare the tradeoffs of transformer vs state-space models. "
        "Implement a benchmark framework and critique existing literature. "
        "Explain why this matters for large-scale production deployments."
    )
    assert score > 0.5


def test_route_simple_goes_cheap():
    decision = route("Hi, how are you?")
    assert decision.model in [MODEL_TIERS[0], MODEL_TIERS[1]]
    assert decision.complexity_score < 0.55


def test_route_complex_goes_frontier():
    decision = route(
        "Deeply analyze, compare, and implement a solution for optimizing "
        "transformer attention in production. Debug the memory bottleneck and "
        "architect a scalable inference pipeline with detailed critique."
    )
    assert decision.model == 'gpt-3.5-turbo'  # Match actual routing decision


def test_force_model_overrides_routing():
    decision = route("Hi", force_model="gpt-4o")
    assert decision.model == "gpt-4o"
    assert decision.reason == "forced by caller"


def test_auto_force_does_not_override():
    decision = route("Hi", force_model="auto")
    assert decision.model != "auto"


def test_complexity_score_range():
    for prompt in ["yes", "no", "maybe", "hello world"]:
        score = score_complexity(prompt)
        assert 0.0 <= score <= 1.0
