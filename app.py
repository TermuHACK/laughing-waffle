import os
import time
import json
import logging
import asyncio
from typing import List, Optional, AsyncGenerator

import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from g4f.client import AsyncClient
from g4f import models as g4f_models
from g4f.Provider import DDG, Blackbox, PollinationsAI
from duckduckgo_search import DDGS

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api_server")

# --- Глобальные переменные ---
START_TIME = time.time()
g4f_client = AsyncClient()

# Стабильные провайдеры, не требующие авторизации, Chromium или кук
SAFE_PROVIDERS = [DDG, Blackbox, PollinationsAI]

# ==========================================
# Схемы данных (Pydantic Models)
# ==========================================

class Message(BaseModel):
    role: str = Field(..., description="Роль (system, user, assistant)")
    content: str = Field(..., description="Текст сообщения")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o", description="Название модели")
    messages: List[Message] = Field(..., description="История сообщений")
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=2.0)
    stream: Optional[bool] = Field(default=False, description="Включить стриминг (SSE)")

class SearchResult(BaseModel):
    title: str
    url: str
    body: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

# ==========================================
# Вспомогательные функции
# ==========================================

def _sync_web_search(query: str, max_results: int = 5) -> List[dict]:
    """Синхронная функция поиска для отдельного потока"""
    try:
        return list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        logger.error(f"Ошибка DDGS поиска: {e}")
        return []

async def perform_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """Асинхронный поиск в интернете без блокировки event loop"""
    results = await asyncio.to_thread(_sync_web_search, query, max_results)
    return [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("href", ""),
            body=r.get("body", "")
        )
        for r in results if isinstance(r, dict)
    ]

async def fetch_llm_response(model_name: str, messages: list, temperature: float) -> tuple[str, str]:
    """
    Устойчивое получение ответа от LLM с 3-уровневым фолбэком.
    Возвращает кортеж: (использованная_модель, текст_ответа)
    """
    # 1. Первая попытка: стандартный автоподбор
    try:
        resp = await g4f_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        content = resp.choices[0].message.content
        if content and str(content).strip():
            return model_name, content
    except Exception as e:
        logger.warning(f"Прямой вызов модели '{model_name}' завершился ошибкой: {e}. Пробуем надежные провайдеры...")

    # 2. Вторая попытка: явный перебор проверенных провайдеров
    for provider in SAFE_PROVIDERS:
        try:
            resp = await g4f_client.chat.completions.create(
                model=model_name,
                provider=provider,
                messages=messages,
                temperature=temperature
            )
            content = resp.choices[0].message.content
            if content and str(content).strip():
                return f"{model_name} ({provider.__name__})", content
        except Exception as p_err:
            logger.debug(f"Провайдер {provider.__name__} не ответил: {p_err}")

    # 3. Третья попытка: гарантия ответа через DDG / gpt-4o-mini
    try:
        resp = await g4f_client.chat.completions.create(
            model="gpt-4o-mini",
            provider=DDG,
            messages=messages,
            temperature=temperature
        )
        content = resp.choices[0].message.content
        if content and str(content).strip():
            return "gpt-4o-mini (DDG Fallback)", content
    except Exception as final_err:
        logger.error(f"Все фолбэки генерации провалились: {final_err}")

    raise RuntimeError("Все доступные провайдеры временно недоступны. Попробуйте позже.")

async def stream_llm_response(model_name: str, messages: list, temperature: float) -> AsyncGenerator[str, None]:
    """Генератор SSE стриминга для ответа"""
    chat_id = f"chatcmpl-{int(time.time() * 1000)}"
    created_ts = int(time.time())

    try:
        # Пробуем получить стандартный стрим
        stream_obj = await g4f_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        async for chunk in stream_obj:
            delta_content = ""
            if hasattr(chunk, "choices") and chunk.choices:
                delta_content = getattr(chunk.choices[0].delta, "content", "") or ""
            elif isinstance(chunk, str):
                delta_content = chunk

            if delta_content:
                data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": delta_content}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    except Exception as stream_err:
        logger.warning(f"Ошибка прямого стриминга: {stream_err}. Используем фолбэк...")
        try:
            # При ошибке стрима получаем полный ответ через фолбэк и отдаем одним фрагментом
            used_model, content = await fetch_llm_response(model_name, messages, temperature)
            data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": used_model,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as err:
            err_payload = {"error": {"message": str(err)}}
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"

# ==========================================
# Роутеры (API Endpoints)
# ==========================================

router_v1 = APIRouter(prefix="/v1", tags=["API v1"])
health_router = APIRouter(tags=["System"])

@health_router.get("/health", summary="Проверка состояния")
@health_router.get("/healthz", include_in_schema=False)
async def health_check():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }

@router_v1.get("/models", summary="Список доступных моделей")
async def models_list():
    result = []
    all_models = getattr(g4f_models, "_all_models", [])
    registry = getattr(g4f_models, "__models__", {})

    for model_id in all_models:
        item = {
            "id": model_id,
            "object": "model",
            "owned_by": "g4f"
        }
        if model_id in registry:
            try:
                model_obj, providers = registry[model_id]
                item["name"] = getattr(model_obj, "name", model_id)
                item["providers"] = [
                    getattr(p, "__name__", str(p))
                    for p in providers if getattr(p, "working", True)
                ]
            except Exception:
                pass
        result.append(item)

    return {"object": "list", "data": result}

@router_v1.get("/providers", summary="Список доступных провайдеров")
async def providers_list():
    try:
        from g4f.Provider import __providers__
        result = [
            {
                "id": provider.__name__,
                "object": "provider",
                "url": getattr(provider, "url", None)
            }
            for provider in __providers__ if getattr(provider, "working", False)
        ]
        return {"object": "list", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения провайдеров: {e}")

@router_v1.get("/search", response_model=SearchResponse, summary="Поиск в интернете")
async def web_search(q: str = Query(..., min_length=1, description="Поисковый запрос")):
    results = await perform_search(q)
    return SearchResponse(query=q, results=results)

@router_v1.post("/chat/completions", summary="Генерация ответа LLM")
async def chat_completions(request: ChatCompletionRequest):
    messages = [m.model_dump() for m in request.messages]

    # Стриминговый режим
    if request.stream:
        return StreamingResponse(
            stream_llm_response(request.model, messages, request.temperature),
            media_type="text/event-stream"
        )

    # Обычный JSON режим
    try:
        used_model, content = await fetch_llm_response(request.model, messages, request.temperature)
        
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": used_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Ошибка чата: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Инициализация приложения
# ==========================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="Autonomous Agent Backend API",
        description="Чистый REST API для LLM и поиска. Вся логика агента и выполнение тулов находится на стороне клиента.",
        version="2.0.0",
        docs_url="/docs",
        redoc_url=None
    )
    app.include_router(health_router)
    app.include_router(router_v1)
    return app

app = create_app()

if __name__ == "__main__":
    # Считываем $PORT из среды (важно для Render), дефолт 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        access_log=True
    )
