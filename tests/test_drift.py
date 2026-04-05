"""Tests for the drift detector."""
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from drift.detector import _jensen_shannon_divergence, _cluster_distribution, DriftResult


def test_jsd_identical_distributions():
    p = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    jsd = _jensen_shannon_divergence(p, p.copy())
    assert jsd == pytest.approx(0.0, abs=1e-6)


def test_jsd_different_distributions():
    p = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    jsd = _jensen_shannon_divergence(p, q)
    assert jsd > 0.5


def test_jsd_output_range():
    rng = np.random.default_rng(42)
    for _ in range(20):
        p = rng.dirichlet(np.ones(10))
        q = rng.dirichlet(np.ones(10))
        jsd = _jensen_shannon_divergence(p, q)
        assert 0.0 <= jsd <= 1.0 + 1e-6


def test_cluster_distribution_shape():
    embeddings_2d = np.random.randn(100, 2)
    labels = np.zeros(100, dtype=int)
    dist = _cluster_distribution(embeddings_2d, labels, n_bins=10)
    assert dist.shape == (100,)  # 10*10 bins flattened
    assert np.all(dist > 0)  # Laplace smoothing ensures no zeros


def test_drift_result_dataclass():
    result = DriftResult(score=0.12, is_drift=False, details={"n_samples": 100})
    assert result.score == 0.12
    assert not result.is_drift
    assert result.details["n_samples"] == 100
