import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from g4f.client import AsyncClient
from duckduckgo_search import DDGS

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
g4f_client = AsyncClient()
START_TIME = time.time()

SYSTEM_TOOL_INSTRUCTION = """Ты — автономный терминальный ассистент с доступом к локальной системе пользователя.
Тебе доступны инструменты для локального выполнения:

1. Выполнение Bash-команд:
```bash
ваша_команда

```

2. Чтение файла:

```read_file
путь/к/файлу

```

3. Запись/создание файла:

```write_file
путь/к/файлу
содержимое файла

```

После вызова инструмента клиент выполнит его и передаст тебе ПОЛНЫЙ вывод (stdout и stderr). Оценивай результат и продолжай решение задачи.
"""

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
web_search: Optional[bool] = False

@app.get("/health")
@app.get("/healthz")
async def health():
return {"status": "ok", "uptime_seconds": round(time.time() - START_TIME, 2)}

@app.get("/v1/models")
async def list_models():
return {
"object": "list",
"data": [
{"id": "gpt-4o", "object": "model", "owned_by": "g4f"},
{"id": "gpt-4o-mini", "object": "model", "owned_by": "g4f"},
{"id": "llama-3.3-70b", "object": "model", "owned_by": "g4f"},
{"id": "deepseek-v3", "object": "model", "owned_by": "g4f"},
{"id": "qwen-2.5-72b", "object": "model", "owned_by": "g4f"}
]
}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
if request.stream:
raise HTTPException(status_code=400, detail="Streaming non-supported")

```
try:
    raw_messages = [m.model_dump() for m in request.messages]

    if not any(m.get("role") == "system" for m in raw_messages):
        raw_messages.insert(0, {"role": "system", "content": SYSTEM_TOOL_INSTRUCTION})

    if request.web_search and len(raw_messages) > 0:
        last_user_prompt = raw_messages[-1]["content"]
        search_context = search_ddg(last_user_prompt)
        raw_messages.insert(1, {
            "role": "system",
            "content": f"Свежие результаты поиска из DuckDuckGo:\n{search_context}"
        })

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
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
    }
except Exception as e:
    raise HTTPException(status_code=500, detail=f"G4F Execution Error: {str(e)}")

```

if **name** == "**main**":
uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
