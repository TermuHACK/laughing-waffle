import time
import logging
import asyncio
from typing import List, Optional, Any

import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from g4f.client import AsyncClient
from g4f import models as g4f_models
from duckduckgo_search import DDGS

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Инициализация глобальных переменных ---
START_TIME = time.time()
g4f_client = AsyncClient()

# ==========================================
# Схемы данных (Pydantic Models)
# ==========================================

class Message(BaseModel):
    role: str = Field(..., description="Роль автора сообщения (system, user, assistant)")
    content: str = Field(..., description="Текст сообщения")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o", description="Название модели")
    messages: List[Message] = Field(..., description="История сообщений, включая системный промпт")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    stream: bool = Field(default=False, description="Поддержка стриминга (на данный момент отключена)")

class SearchResult(BaseModel):
    title: str
    url: str
    body: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

# ==========================================
# Сервисные функции
# ==========================================

def _sync_web_search(query: str, max_results: int = 5) -> List[dict]:
    """Синхронная функция поиска для выполнения в отдельном потоке"""
    try:
        return list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        logger.error(f"Ошибка duckduckgo_search: {e}")
        raise ValueError(str(e))

async def perform_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """Асинхронная обертка для поиска, чтобы не блокировать event loop FastAPI"""
    try:
        results = await asyncio.to_thread(_sync_web_search, query, max_results)
        
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                body=r.get("body", "")
            )
            for r in results if r
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {e}")

# ==========================================
# Роутеры (API Endpoints)
# ==========================================

router_v1 = APIRouter(prefix="/v1", tags=["API v1"])
health_router = APIRouter(tags=["System"])

@health_router.get("/health", summary="Проверка состояния сервера")
@health_router.get("/healthz", include_in_schema=False)
async def health_check():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }

@router_v1.get("/models", summary="Получить список доступных моделей")
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
            except Exception as e:
                logger.debug(f"Ошибка парсинга модели {model_id}: {e}")
                
        result.append(item)

    return {"object": "list", "data": result}

@router_v1.get("/providers", summary="Получить список доступных провайдеров")
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
        logger.error(f"Ошибка получения провайдеров: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения списка провайдеров")

@router_v1.get("/search", response_model=SearchResponse, summary="Поиск в интернете")
async def web_search(q: str = Query(..., min_length=1, description="Поисковый запрос")):
    """
    Выполняет поиск через DuckDuckGo.
    Возвращает структурированный JSON с результатами, который клиент может использовать как tool.
    """
    results = await perform_search(q)
    return SearchResponse(query=q, results=results)

@router_v1.post("/chat/completions", summary="Генерация ответа LLM")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-совместимый эндпоинт для генерации текста.
    Системный промпт и логика инструментов (tools) должны быть переданы клиентом в массиве messages.
    """
    if request.stream:
        raise HTTPException(status_code=400, detail="Стриминг временно не поддерживается")

    try:
        response = await g4f_client.chat.completions.create(
            model=request.model,
            messages=[m.model_dump() for m in request.messages],
            temperature=request.temperature
        )
        content = response.choices[0].message.content

        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
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
        logger.error(f"Ошибка генерации G4F: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации ответа: {e}")

# ==========================================
# Инициализация приложения
# ==========================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="Autonomous Agent Backend API",
        description="REST API для работы агента (LLM + Search). Вся логика выполнения команд лежит на клиенте.",
        version="1.0.0"
    )
    
    app.include_router(health_router)
    app.include_router(router_v1)
    
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        access_log=True
                               )
