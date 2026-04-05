"""
Semantic Drift Detector

Pipeline:
  1. Embed recent LLM responses with sentence-transformers
  2. Reduce to 2D with UMAP
  3. Cluster with HDBSCAN
  4. Compare cluster distribution to baseline (stored in Redis / file)
  5. Alert if Jensen-Shannon divergence exceeds threshold

Design choice: UMAP + HDBSCAN over simpler centroid drift because it handles
multi-modal output distributions (e.g. the model suddenly starts producing a
new response pattern that the centroid approach would miss).
"""
import json
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import structlog

from app.config import get_settings
from drift.embedder import embed

logger = structlog.get_logger()
settings = get_settings()

BASELINE_PATH = "/tmp/sentinel_baseline.pkl"


@dataclass
class DriftResult:
    score: float          # JS divergence (0 = no drift, 1 = complete drift)
    is_drift: bool
    details: dict = field(default_factory=dict)


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute JS divergence between two probability distributions."""
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    # Clip to avoid log(0)
    kl_pm = np.sum(np.where(p > 0, p * np.log(p / np.clip(m, 1e-10, None)), 0))
    kl_qm = np.sum(np.where(q > 0, q * np.log(q / np.clip(m, 1e-10, None)), 0))
    return float(0.5 * (kl_pm + kl_qm))


def _cluster_distribution(embeddings_2d: np.ndarray, labels: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Bin the 2D UMAP space into a histogram as a proxy for cluster distribution."""
    hist, _, _ = np.histogram2d(
        embeddings_2d[:, 0], embeddings_2d[:, 1], bins=n_bins,
        range=[[-15, 15], [-15, 15]]
    )
    hist = hist.flatten() + 1e-6  # Laplace smoothing
    return hist


def detect_drift(texts: list[str]) -> DriftResult:
    """
    Detect semantic drift in a list of LLM response texts.
    On first run, establishes a baseline. Subsequent runs compare against it.
    """
    try:
        import umap
        import hdbscan
    except ImportError:
        logger.error("drift_deps_missing", msg="Install umap-learn and hdbscan")
        return DriftResult(score=0.0, is_drift=False, details={"error": "deps missing"})

    logger.info("drift_embedding_start", n_texts=len(texts))
    embeddings = embed(texts)

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
    labels = clusterer.fit_predict(embeddings_2d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    current_dist = _cluster_distribution(embeddings_2d, labels)

    if not os.path.exists(BASELINE_PATH):
        logger.info("drift_baseline_created", n_clusters=n_clusters)
        with open(BASELINE_PATH, "wb") as f:
            pickle.dump({"distribution": current_dist, "n_clusters": n_clusters}, f)
        return DriftResult(score=0.0, is_drift=False, details={"status": "baseline_created"})

    with open(BASELINE_PATH, "rb") as f:
        baseline = pickle.load(f)

    jsd = _jensen_shannon_divergence(baseline["distribution"], current_dist)
    is_drift = jsd > settings.drift_threshold

    logger.info("drift_check_complete", jsd=round(jsd, 4), threshold=settings.drift_threshold, is_drift=is_drift)

    return DriftResult(
        score=round(jsd, 4),
        is_drift=is_drift,
        details={
            "jsd": jsd,
            "n_clusters_current": n_clusters,
            "n_clusters_baseline": baseline.get("n_clusters"),
            "n_samples": len(texts),
        },
    )
