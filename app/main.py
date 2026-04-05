import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.config import get_settings
from app.database import init_db
from app.api.proxy import router as proxy_router
from app.api.prompts import router as prompts_router
from monitoring.metrics import setup_metrics

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LLM Ops Sentinel", env=settings.app_env)
    await init_db()
    setup_metrics()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="LLM Ops Sentinel",
    description="Production LLM observability, evaluation, and self-healing prompt versioning",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Routers
app.include_router(proxy_router, prefix="/v1", tags=["proxy"])
app.include_router(prompts_router, prefix="/prompts", tags=["prompts"])


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}
