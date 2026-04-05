"""
LLM Proxy Router

Intercepts all LLM calls, logs them to Postgres, routes to the optimal model,
and enqueues async evaluation via Celery.
"""
import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cost import calculate_cost
from app.core.hasher import hash_prompt
from app.core.router import route
from app.database import LLMCall, get_db
from monitoring.metrics import (
    llm_calls_total,
    llm_latency_seconds,
    llm_cost_dollars,
    llm_tokens_total,
)
from workers.tasks import evaluate_response

logger = structlog.get_logger()
settings = get_settings()
router = APIRouter()


class ChatRequest(BaseModel):
    prompt: str
    model: str = "auto"  # "auto" triggers cost-aware routing
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.7
    metadata: dict[str, Any] = {}


class ChatResponse(BaseModel):
    id: str
    model: str
    response: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    complexity_score: float
    routing_reason: str


async def _call_openai(prompt: str, system: str | None, model: str, max_tokens: int, temperature: float) -> dict:
    """Make an async call to the OpenAI API."""
    import openai
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "text": resp.choices[0].message.content,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }


async def _call_anthropic(prompt: str, system: str | None, model: str, max_tokens: int, temperature: float) -> dict:
    """Make an async call to the Anthropic API."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                  messages=[{"role": "user", "content": prompt}])
    if system:
        kwargs["system"] = system

    resp = await client.messages.create(**kwargs)
    return {
        "text": resp.content[0].text,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


async def _dispatch(model: str, prompt: str, system: str | None, max_tokens: int, temperature: float) -> dict:
    """Dispatch to the right SDK based on model name."""
    if model.startswith("claude"):
        return await _call_anthropic(prompt, system, model, max_tokens, temperature)
    return await _call_openai(prompt, system, model, max_tokens, temperature)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    routing = route(req.prompt, force_model=req.model)
    model = routing.model

    start = time.perf_counter()
    try:
        result = await _dispatch(model, req.prompt, req.system, req.max_tokens, req.temperature)
    except Exception as exc:
        logger.error("llm_call_failed", model=model, error=str(exc))
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    cost = calculate_cost(model, result["input_tokens"], result["output_tokens"])
    prompt_hash = hash_prompt(req.prompt)

    # Persist to DB
    call = LLMCall(
        prompt_hash=prompt_hash,
        model=model,
        prompt_text=req.prompt,
        response_text=result["text"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        cost_usd=cost,
        latency_ms=latency_ms,
        metadata_=req.metadata,
    )
    db.add(call)
    await db.commit()
    await db.refresh(call)

    # Prometheus metrics
    llm_calls_total.labels(model=model).inc()
    llm_latency_seconds.labels(model=model).observe(latency_ms / 1000)
    llm_cost_dollars.labels(model=model).inc(cost)
    llm_tokens_total.labels(model=model, type="input").inc(result["input_tokens"])
    llm_tokens_total.labels(model=model, type="output").inc(result["output_tokens"])

    # Enqueue async evaluation (non-blocking)
    evaluate_response.delay(
        call_id=call.id,
        prompt=req.prompt,
        response=result["text"],
        model=model,
    )

    logger.info(
        "llm_call_complete",
        call_id=call.id,
        model=model,
        latency_ms=latency_ms,
        cost_usd=cost,
        complexity=routing.complexity_score,
    )

    return ChatResponse(
        id=call.id,
        model=model,
        response=result["text"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        cost_usd=cost,
        latency_ms=latency_ms,
        complexity_score=routing.complexity_score,
        routing_reason=routing.reason,
    )
