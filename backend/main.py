"""
PhilTrace Backend - FastAPI entrypoint.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.analysis_engine import get_engine
from backend.config import settings
from backend.data_models import AnalyzeRequest, AnalyzeResponse, ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from backend.llm_service import ChatMessage, OllamaService, OllamaServiceError, check_ollama_status, get_philosophy_system_prompt
from backend.logging_utils import get_logger
from backend.sense_interpreter import (
    DeepSeekInterpreter,
    SenseInterpretation,
    create_interpreter,
)

logger = get_logger(__name__)
VERSION = "1.1.0"


def generate_request_id() -> str:
    return str(uuid.uuid4())[:8]


def error_payload(error: str, message: str, request_id: str, detail: str | None = None) -> Dict[str, Any]:
    payload = ErrorResponse(error=error, message=message, detail=detail, request_id=request_id)
    return payload.model_dump()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine_boot_error = None

    if settings.bootstrap_engine_on_startup:
        try:
            get_engine()
            logger.info("Analysis engine bootstrapped on startup")
        except Exception as exc:  # noqa: BLE001
            app.state.engine_boot_error = str(exc)
            logger.exception("Engine bootstrap failed: %s", exc)
    else:
        logger.info("Engine bootstrap skipped by configuration")

    yield


app = FastAPI(
    title="PhilTrace API",
    description="Backend API for philosophy concept drift analysis",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = generate_request_id()
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.exception("request_id=%s path=%s method=%s failed_in=%dms", request_id, request.url.path, request.method, elapsed_ms)
        raise

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s path=%s method=%s status=%s duration_ms=%d",
        request_id,
        request.url.path,
        request.method,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    return {"name": "PhilTrace Backend API", "version": VERSION, "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        gpu_available = False

    engine_boot_error = getattr(request.app.state, "engine_boot_error", None)

    model_loaded = False
    corpus_loaded = False
    runtime_error = None
    try:
        engine = get_engine()
        model_loaded = bool(engine.model is not None)
        corpus_loaded = any(len(files) > 0 for files in getattr(engine, "corpus_files", {}).values())
    except Exception as exc:  # noqa: BLE001
        runtime_error = str(exc)

    ollama_status = check_ollama_status()
    status = "healthy" if runtime_error is None else "degraded"

    return HealthResponse(
        status=status,
        version=VERSION,
        gpu_available=gpu_available,
        model_loaded=model_loaded,
        corpus_loaded=corpus_loaded,
        extra={
            "request_id": request.state.request_id,
            "ollama_available": ollama_status.get("available", False),
            "ollama_models_count": ollama_status.get("count", 0),
            "engine_boot_error": engine_boot_error,
            "engine_runtime_error": runtime_error,
        },
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_concept(request: Request, payload: AnalyzeRequest) -> AnalyzeResponse:
    request_id = request.state.request_id

    try:
        engine = get_engine()
        result = engine.analyze(
            word=payload.word,
            umap_params=payload.umap_params,
            hdbscan_params=payload.hdbscan_params,
            force_recompute=payload.force_recompute,
        )
    except ValueError as exc:
        logger.warning("request_id=%s analyze validation failed: %s", request_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("request_id=%s analyze failed: %s", request_id, exc)
        raise HTTPException(status_code=500, detail="Analysis failed") from exc

    return AnalyzeResponse(
        concept=result["concept"],
        periods=result["periods"],
        definitions=result["definitions"],
        period_years=result["period_years"],
        scatter_data=result["scatter_data"],
        drift_data=result["drift_data"],
        cluster_stats=result["cluster_stats"],
        quotations=result["quotations"],
        confidence_score=result["confidence_score"],
        analysis_timestamp=result["analysis_timestamp"],
        processing_time_ms=result["processing_time_ms"],
        model_info=result.get("model_info"),
    )


# ============================================================================
# Interpretation endpoint
# ============================================================================

@app.post("/interpret", tags=["Interpretation"])
async def interpret_clusters(request: Request) -> Dict[str, Any]:
    """
    Run LLM sense interpretation on the clustered results of a concept.

    Request body:
        word: str - target concept
        provider: str - "ollama" | "deepseek" (default from config)
        api_key: str | null - required for deepseek
        force_recompute: bool - ignore cache
    """
    request_id = request.state.request_id
    body = await request.json()
    word = body.get("word", "").strip()
    provider = body.get("provider")  # None means use config default
    api_key = body.get("api_key")
    force_recompute = body.get("force_recompute", False)

    if not word:
        raise HTTPException(status_code=400, detail="word is required")

    try:
        # Get clustered data from engine
        engine = get_engine()
        df = engine.get_vectors(word)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for '{word}'")

        clustered_df = engine.cluster(df)
        if "label" not in clustered_df.columns:
            raise HTTPException(status_code=500, detail="Clustering failed")

        # Create interpreter with specified provider
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key

        interpreter = create_interpreter(provider=provider, **kwargs)

        # If force_recompute, disable cache temporarily
        if force_recompute:
            interpreter.config.enable_cache = False

        # Run interpretation
        interpretations = interpreter.interpret_all_clusters(word, clustered_df)

        return {
            "word": word,
            "provider": provider or interpreter.config.provider,
            "model": interpreter.get_model_name(),
            "n_clusters": len(interpretations),
            "interpretations": [interp.to_dict() for interp in interpretations],
            "request_id": request_id,
        }

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("request_id=%s interpret failed: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=f"Interpretation failed: {exc}") from exc


# ============================================================================
# DeepSeek chat endpoint (streaming)
# ============================================================================

@app.post("/chat/deepseek", tags=["Chat"])
async def chat_deepseek_stream(request: Request) -> StreamingResponse:
    """
    Stream chat response from DeepSeek API.

    Request body:
        message: str
        api_key: str - DeepSeek API key
        model: str | null - defaults to config deepseek_model
        system_prompt: str | null
        history: list[{role, content}] | null
    """
    request_id = request.state.request_id
    body = await request.json()
    message = body.get("message", "").strip()
    api_key = body.get("api_key", "")
    model = body.get("model")
    system_prompt = body.get("system_prompt") or get_philosophy_system_prompt(enable=True)
    history = body.get("history")

    if not message:
        return JSONResponse(status_code=400, content=error_payload("ValidationError", "message is required", request_id))

    if not api_key:
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return JSONResponse(status_code=400, content=error_payload("ConfigError", "DeepSeek API key not provided", request_id))

    async def generate_deepseek_stream() -> AsyncIterator[str]:
        try:
            interpreter = DeepSeekInterpreter(api_key=api_key, model=model)

            # Build full message with history
            full_message = ""
            if history:
                for item in history:
                    if item.get("role") == "user":
                        full_message += f"User: {item['content']}\n"
                    elif item.get("role") == "assistant":
                        full_message += f"Assistant: {item['content']}\n"
                full_message += f"User: {message}"
            else:
                full_message = message

            for chunk in interpreter.chat_stream(full_message, system_prompt=system_prompt):
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"

            yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("request_id=%s deepseek chat stream failed: %s", request_id, exc)
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"

    return StreamingResponse(generate_deepseek_stream(), media_type="text/event-stream")


@app.get("/models", tags=["Resources"])
async def list_models(request: Request) -> Dict[str, Any]:
    status = check_ollama_status()
    status["request_id"] = request.state.request_id
    if not status.get("available", False):
        status.setdefault("message", "Ollama service is unavailable")
    return status


def _validate_chat_model(request_model: str, available_models: List[Dict[str, Any]]) -> None:
    available_names = [model.get("name") for model in available_models]
    if request_model not in available_names:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ModelNotAvailable",
                "message": f"Model '{request_model}' is not available",
                "available_models": available_names,
            },
        )


@app.post("/chat", tags=["Chat"])
async def chat_stream(request: Request, payload: ChatRequest):
    request_id = request.state.request_id
    status = check_ollama_status()
    if not status.get("available", False):
        return JSONResponse(status_code=503, content=error_payload("ServiceUnavailable", "Ollama service unavailable", request_id, status.get("error")))

    _validate_chat_model(payload.model, status.get("models", []))

    messages: List[ChatMessage] = []
    if payload.adapter:
        system_prompt = get_philosophy_system_prompt(enable=True)
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))

    if payload.history:
        messages.extend(ChatMessage(role=item.role, content=item.content) for item in payload.history)

    messages.append(ChatMessage(role="user", content=payload.message))

    async def generate_stream() -> AsyncIterator[str]:
        try:
            with OllamaService() as service:
                for chunk in service.chat_stream(
                    model=payload.model,
                    messages=messages,
                    system_prompt=get_philosophy_system_prompt(enable=bool(payload.adapter)),
                    stream=True,
                ):
                    yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"

            yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("request_id=%s chat stream failed: %s", request_id, exc)
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.post("/chat/simple", response_model=ChatResponse, tags=["Chat"])
async def chat_simple(request: Request, payload: ChatRequest) -> ChatResponse:
    request_id = request.state.request_id
    started = time.time()

    status = check_ollama_status()
    if not status.get("available", False):
        raise HTTPException(status_code=503, detail="Ollama service unavailable")

    _validate_chat_model(payload.model, status.get("models", []))

    try:
        with OllamaService() as service:
            response_text = service.chat(
                model=payload.model,
                message=payload.message,
                system_prompt=get_philosophy_system_prompt(enable=bool(payload.adapter)),
            )
    except OllamaServiceError as exc:
        logger.warning("request_id=%s chat unavailable: %s", request_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("request_id=%s chat failed: %s", request_id, exc)
        raise HTTPException(status_code=500, detail="Chat failed") from exc

    elapsed_ms = int((time.time() - started) * 1000)
    return ChatResponse(
        response=response_text,
        model=payload.model,
        adapter=payload.adapter,
        token_count=max(1, len(response_text) // 4),
        response_time_ms=elapsed_ms,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", generate_request_id())
    logger.warning("request_id=%s validation error: %s", request_id, exc)
    return JSONResponse(
        status_code=422,
        content=error_payload("ValidationError", "Invalid request payload", request_id, detail=str(exc)),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", generate_request_id())
    if isinstance(exc.detail, dict):
        content = {"request_id": request_id, **exc.detail}
    else:
        content = error_payload("HTTPError", str(exc.detail), request_id)
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", generate_request_id())
    logger.exception("request_id=%s unhandled exception: %s", request_id, exc)
    return JSONResponse(
        status_code=500,
        content=error_payload("InternalError", "Internal server error", request_id, detail=str(exc)),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=settings.reload, log_level=settings.log_level.lower())
