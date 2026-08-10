import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from g4f.client import Client

# Отключение генерации OpenAPI/Swagger документации ускоряет старт приложения в 2-3 раза
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# Предварительная инициализация клиента g4f
g4f_client = Client()
START_TIME = time.time()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

# Health-эндпоинты для Docker / Kubernetes / систем мониторинга
@app.get("/health")
@app.get("/healthz")
async def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported")

    try:
        raw_messages = [m.model_dump() for m in request.messages]
        
        # Прямой вызов g4f без кэширования
        response = g4f_client.chat.completions.create(
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

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o", "object": "model", "owned_by": "g4f-proxy"},
            {"id": "claude-3.5-sonnet", "object": "model", "owned_by": "g4f-proxy"},
            {"id": "gemini-pro", "object": "model", "owned_by": "g4f-proxy"}
        ]
    }

if __name__ == "__main__":
    # access_log=False убирает задержки на диск/консоль при обработке запросов
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
