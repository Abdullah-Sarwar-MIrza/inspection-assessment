"""Bounded image upload API. Launch with uvicorn casting_inspector.api:app."""
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .model import MAX_BYTES, Predictor, decode_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("casting_inspector")


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(max(1, int(os.getenv("TORCH_THREADS", "2"))))
    app.state.predictor = Predictor(os.getenv("MODEL_DIR", "artifacts"))
    app.state.gate = asyncio.Semaphore(1)
    logger.info("model_loaded version=%s", app.state.predictor.metadata["model_version"])
    yield
    app.state.predictor = None


app = FastAPI(title="Casting Defect Inspector", version="1.0.0", lifespan=lifespan)


class UploadBodyLimit:
    """Bound raw body before multipart parsing, including chunked uploads."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        limit = MAX_BYTES + 65536  # small allowance for multipart headers
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > limit:
                from starlette.responses import JSONResponse
                logger.warning("upload_rejected reason=body_limit bytes=%s", total)
                return await JSONResponse(status_code=413, content={"detail": "Request body too large"})(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()
        await self.app(scope, bounded_receive, send)


app.add_middleware(UploadBodyLimit)


class Prediction(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    defect_probability: float = Field(ge=0, le=1)
    threshold: float
    model_version: str
    request_id: str
    latency_ms: float


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed request_id=%s", request.state.request_id)
        from fastapi.responses import JSONResponse
        response = JSONResponse(status_code=500, content={"detail": "Internal inference error"})
    response.headers["X-Request-ID"] = request.state.request_id
    logger.info("request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
                request.state.request_id, request.method, request.url.path,
                response.status_code, 1000 * (time.perf_counter() - started))
    return response


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready(request: Request):
    if getattr(request.app.state, "predictor", None) is None:
        raise HTTPException(503, "Model unavailable")
    return {"status": "ready", "model_version": request.app.state.predictor.metadata["model_version"]}


@app.post("/predict", response_model=Prediction)
async def predict(request: Request, file: UploadFile = File(...)):
    start = time.perf_counter()
    try:
        if file.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(415, "Use image/jpeg or image/png")
        raw = await file.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, "Image exceeds 8 MiB")
        async with request.app.state.gate:
            try:
                image = await run_in_threadpool(decode_image, raw)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            result = await run_in_threadpool(request.app.state.predictor.predict, image)
        return {**result, "request_id": request.state.request_id,
                "latency_ms": round((time.perf_counter() - start) * 1000, 3)}
    finally:
        await file.close()
