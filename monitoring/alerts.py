"""
Alert senders for drift and rollback events.
Supports Slack webhooks and PagerDuty Events API v2.
"""
import structlog
import requests

from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


def send_drift_alert(drift_result) -> None:
    message = (
        f":warning: *LLM Ops Sentinel - Semantic Drift Detected*\n"
        f"Drift score: {drift_result.score:.4f} (threshold: {settings.drift_threshold})\n"
        f"Samples: {drift_result.details.get('n_samples', 'N/A')}\n"
        f"Clusters (current/baseline): "
        f"{drift_result.details.get('n_clusters_current')} / {drift_result.details.get('n_clusters_baseline')}"
    )
    _post_slack(message)
    _trigger_pagerduty(
        summary="LLM semantic drift detected",
        severity="warning",
        details=drift_result.details,
    )


def send_rollback_alert(prompt_name: str, version: str, avg_score: float) -> None:
    message = (
        f":rotating_light: *LLM Ops Sentinel - Auto Rollback Triggered*\n"
        f"Prompt: {prompt_name} version {version}\n"
        f"Average eval score: {avg_score:.3f} (threshold: {settings.rollback_score_threshold})\n"
        f"The canary has been removed."
    )
    _post_slack(message)
    _trigger_pagerduty(
        summary=f"Prompt rollback: {prompt_name} v{version}",
        severity="error",
        details={"prompt_name": prompt_name, "version": version, "avg_score": avg_score},
    )


def _post_slack(text: str) -> None:
    if not settings.slack_webhook_url:
        logger.debug("slack_not_configured")
        return
    try:
        resp = requests.post(settings.slack_webhook_url, json={"text": text}, timeout=5)
        resp.raise_for_status()
        logger.info("slack_alert_sent")
    except Exception as exc:
        logger.error("slack_alert_failed", error=str(exc))


def _trigger_pagerduty(summary: str, severity: str, details: dict) -> None:
    if not settings.pagerduty_api_key:
        logger.debug("pagerduty_not_configured")
        return
    try:
        payload = {
            "routing_key": settings.pagerduty_api_key,
            "event_action": "trigger",
            "payload": {
                "summary": summary,
                "severity": severity,
                "source": "llm-ops-sentinel",
                "custom_details": details,
            },
        }
        resp = requests.post("https://events.pagerduty.com/v2/enqueue", json=payload, timeout=5)
        resp.raise_for_status()
        logger.info("pagerduty_alert_sent")
    except Exception as exc:
        logger.error("pagerduty_alert_failed", error=str(exc))
