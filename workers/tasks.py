"""
Celery Tasks

- evaluate_response: Score a single LLM response (runs after every call)
- run_drift_detection: Nightly job to detect semantic drift
- check_rollback: Hourly job to roll back prompts with degraded scores
"""
import asyncio
import uuid

import mlflow
import structlog
from celery import Task
from sqlalchemy import select, func, update

from app.config import get_settings
from app.database import AsyncSessionLocal, EvaluationResult, LLMCall, PromptVersion, DriftEvent
from app.evaluators.judge import evaluate
from drift.detector import detect_drift
from monitoring.alerts import send_drift_alert, send_rollback_alert
from workers.celery_app import celery_app

logger = structlog.get_logger()
settings = get_settings()


def run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Evaluation Task ────────────────────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=3, default_retry_delay=30, queue="evaluation")
def evaluate_response(self: Task, call_id: str, prompt: str, response: str, model: str):
    """Evaluate a single LLM response and log scores to MLflow + Postgres."""
    logger.info("eval_task_started", call_id=call_id)

    try:
        result = run_async(evaluate(prompt, response))
    except Exception as exc:
        logger.error("eval_failed", call_id=call_id, error=str(exc))
        raise self.retry(exc=exc)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("llm-eval")

    with mlflow.start_run(run_name=f"eval-{call_id[:8]}") as run:
        mlflow.log_params({"model": model, "call_id": call_id})
        mlflow.log_metrics({
            "faithfulness": result.faithfulness,
            "relevance": result.relevance,
            "toxicity": result.toxicity,
            "overall_score": result.overall_score,
        })
        run_id = run.info.run_id

    async def _save():
        async with AsyncSessionLocal() as db:
            eval_row = EvaluationResult(
                call_id=call_id,
                faithfulness=result.faithfulness,
                relevance=result.relevance,
                toxicity=result.toxicity,
                overall_score=result.overall_score,
                judge_model=result.judge_model,
                mlflow_run_id=run_id,
            )
            db.add(eval_row)
            await db.commit()

    run_async(_save())
    logger.info("eval_task_complete", call_id=call_id, overall=result.overall_score, mlflow_run=run_id)
    return {"call_id": call_id, "overall_score": result.overall_score}


# ── Drift Detection Task ───────────────────────────────────────────────────────

@celery_app.task(bind=True, queue="alerts")
def run_drift_detection(self: Task):
    """Nightly: embed recent responses and compare against golden set."""
    logger.info("drift_detection_started")

    async def _run():
        async with AsyncSessionLocal() as db:
            # Fetch last 500 responses
            result = await db.execute(
                select(LLMCall.response_text)
                .order_by(LLMCall.created_at.desc())
                .limit(500)
            )
            texts = [row[0] for row in result.fetchall()]

        if len(texts) < 20:
            logger.info("drift_skipped_insufficient_data", count=len(texts))
            return

        drift_result = detect_drift(texts)

        async with AsyncSessionLocal() as db:
            event = DriftEvent(
                drift_score=drift_result.score,
                threshold=settings.drift_threshold,
                num_samples=len(texts),
                alerted=drift_result.is_drift,
                details=drift_result.details,
            )
            db.add(event)
            await db.commit()

        if drift_result.is_drift:
            logger.warning("drift_detected", score=drift_result.score)
            send_drift_alert(drift_result)

    run_async(_run())


# ── Rollback Check Task ────────────────────────────────────────────────────────

@celery_app.task(bind=True, queue="alerts")
def check_rollback(self: Task):
    """Hourly: check if any canary prompt versions have degraded scores."""
    logger.info("rollback_check_started")

    async def _run():
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(PromptVersion).where(
                    PromptVersion.is_canary == True,
                    PromptVersion.call_count >= 10,
                )
            )
            canary_versions = result.scalars().all()

            for pv in canary_versions:
                # Compute average score for this prompt version's calls
                score_result = await db.execute(
                    select(func.avg(EvaluationResult.overall_score))
                    .join(LLMCall, LLMCall.id == EvaluationResult.call_id)
                    .where(LLMCall.prompt_version == pv.version)
                )
                avg_score = score_result.scalar() or 0.0

                await db.execute(
                    update(PromptVersion)
                    .where(PromptVersion.id == pv.id)
                    .values(avg_score=avg_score)
                )
                await db.commit()

                if avg_score < settings.rollback_score_threshold:
                    logger.warning(
                        "auto_rollback_triggered",
                        prompt_name=pv.name,
                        version=pv.version,
                        avg_score=avg_score,
                    )
                    send_rollback_alert(pv.name, pv.version, avg_score)

    run_async(_run())
