import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from g4f.client import AsyncClient
from g4f import Model, models
from duckduckgo_search import DDGS

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

g4f_client = AsyncClient()
START_TIME = time.time()

# Функция поиска через DuckDuckGo
def search_ddg(query: str, max_results: int = 4) -> str:
    try:
        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            return "Результаты поиска не найдены."
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"[{i}] {r.get('title')}\nURL: {r.get('href')}\nContent: {r.get('body')}\n")
        return "\n".join(formatted)
    except Exception as e:
        return f"Ошибка поиска DuckDuckGo: {str(e)}"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    web_search: Optional[bool] = False # Включение веб-поиска

@app.get("/health")
@app.get("/healthz")
async def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "engine": "g4f-async-proxy"
    }

@app.get("/v1/models")
async def list_models():
    """Динамический фетчинг актуальных моделей из g4f"""
    try:
        # Список наиболее популярных и поддерживаемых алиасов g4f
        all_models = [
            "gpt-4o", "gpt-4o-mini", "gpt-4", 
            "claude-3.5-sonnet", "llama-3.3-70b", 
            "deepseek-v3", "qwen-2.5-72b", "gemini-pro"
        ]
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "owned_by": "g4f"} for m in all_models]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming non-supported")

    try:
        raw_messages = [m.model_dump() for m in request.messages]

        # Если включен веб-поиск, запрашиваем DDG и внедряем в контекст
        if request.web_search and len(raw_messages) > 0:
            last_user_prompt = raw_messages[-1]["content"]
            search_context = search_ddg(last_user_prompt)
            
            system_instruction = {
                "role": "system",
                "content": f"Используй следующие свежие результаты поиска из DuckDuckGo для ответа на вопрос пользователя:\n\n{search_context}"
            }
            raw_messages.insert(0, system_instruction)

        response = await g4f_client.chat.completions.create(
            model=request.model,
            messages=raw_messages
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
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"G4F Execution Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
