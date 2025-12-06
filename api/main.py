"""FastAPI application for ReAsk"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console
from rich.logging import RichHandler

from .database import init_db
from .routes import agent_eval


# Configure Rich logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("reask.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup"""
    console.print("[bold cyan]🚀 Starting ReAsk API...[/]")
    init_db()
    console.print("[bold green]✅ Database initialized[/]")
    console.print("[dim]─" * 50 + "[/]")
    yield
    console.print("\n[bold yellow]👋 Shutting down ReAsk API[/]")


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

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    
    # Skip logging for health checks
    if request.url.path != "/api/health":
        status_color = "green" if response.status_code < 400 else "yellow" if response.status_code < 500 else "red"
        method_color = {
            "GET": "cyan",
            "POST": "green", 
            "PUT": "yellow",
            "DELETE": "red",
            "PATCH": "magenta",
        }.get(request.method, "white")
        console.print(
            f"[bold {method_color}]{request.method}[/] {request.url.path} → [{status_color}]{response.status_code}[/] [dim]({duration:.0f}ms)[/]"
        )
    
    return response


# Include routers
app.include_router(agent_eval.router, prefix="/api", tags=["agent"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "reask-api"}

