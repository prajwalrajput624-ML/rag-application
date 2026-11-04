import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.rag import router_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rag_api")

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
FALLBACK_MESSAGE = "\nSorry, Please Try again"

rag_system: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if router_engine is None:
        raise RuntimeError("router_engine is not initialized")
    rag_system["rag_system"] = router_engine
    logger.info("RAG system loaded")
    yield
    rag_system.clear()
    logger.info("RAG system released")


class QueryRequest(BaseModel):
    query_str: str = Field(
        min_length=1,
        max_length=500,
        description="Ask anything about Maharana Pratap and Mewar history",
        examples=["tell me maharana pratap born year"],
    )


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="RAG API",
    lifespan=lifespan,
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type"],
    )


def stream_tokens(response):
    """Sync generator. Starlette runs it in a threadpool, so the event loop is not blocked."""
    try:
        for token in response.response_gen:
            yield token
    except Exception:
        logger.exception("Streaming failed mid-response")
        yield FALLBACK_MESSAGE


@app.get("/", status_code=status.HTTP_200_OK)
def home():
    return {"message": "Welcome to RAG System"}


@app.get("/health")
def health():
    ready = rag_system.get("rag_system") is not None
    return JSONResponse(
        content={"status": "ok" if ready else "unavailable"},
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.post("/ask")
@limiter.limit("5/minute")
async def ask_rag(payload: QueryRequest, request: Request):
    pipeline = rag_system.get("rag_system")
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is not available",
        )

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(pipeline.query, payload.query_str),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Query timed out: %r", payload.query_str[:100])
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out, please try again",
        )
    except Exception:
        logger.exception("Query failed: %r", payload.query_str[:100])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )

    return StreamingResponse(
        stream_tokens(response),
        media_type="text/plain; charset=utf-8",
    )
