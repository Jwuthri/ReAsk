"""FastAPI application for ReAsk"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .routes import datasets, evaluate


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup"""
    init_db()
    yield


app = FastAPI(
    title="ReAsk API",
    description="API for evaluating LLM conversation datasets",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(evaluate.router, prefix="/api", tags=["evaluate"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "reask-api"}

