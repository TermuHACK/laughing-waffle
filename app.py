import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from g4f.client import AsyncClient
from duckduckgo_search import DDGS
from g4f import models


app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

client = AsyncClient()
START_TIME = time.time()


SYSTEM_PROMPT = r"""
Ты автономный Linux-агент.

Ты можешь выполнять действия на компьютере пользователя через инструменты.

ИНСТРУМЕНТЫ:

Bash:

<<<TOOL bash>>>
команда
<<<END_TOOL>>>

Чтение файла:

<<<TOOL read_file>>>
/path/to/file
<<<END_TOOL>>>

Запись файла:

<<<TOOL write_file>>>
/path/to/file
содержимое
<<<END_TOOL>>>

Поиск текста:

<<<TOOL grep>>>
-rni "текст" /path
<<<END_TOOL>>>

Интернет:

<<<TOOL web_search>>>
поисковый запрос
<<<END_TOOL>>>


ПРАВИЛА:

- После каждого инструмента анализируй результат.
- Если задача требует нескольких действий — выполняй их последовательно.
- Не говори, что действие выполнено, пока не получил результат.
- Не выдумывай результаты команд.
- Используй grep для поиска по проектам.
- Используй read_file для чтения файлов.
- Используй write_file для изменения файлов.
- Используй bash для запуска программ, тестов и диагностики.
- Используй web_search для актуальной информации.
- После изменения файлов проверяй результат.
- Если команда завершилась ошибкой — попробуй разобраться и исправить её.
- Продолжай работу до фактического завершения задачи.
- В конце дай краткий итог.

Ты не просто отвечаешь пользователю — ты самостоятельно решаешь поставленную задачу.
"""


class Message(BaseModel):
    role: str
    content: str


class Request(BaseModel):
    model: str = "gpt-4o"
    messages: List[Message]
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False


def search(query: str) -> str:
    try:
        results = list(
            DDGS().text(
                query,
                max_results=5
            )
        )

        if not results:
            return "Ничего не найдено."

        out = []

        for i, r in enumerate(results, 1):
            out.append(
                f"[{i}] {r.get('title', '')}\n"
                f"URL: {r.get('href', '')}\n"
                f"{r.get('body', '')}"
            )

        return "\n\n".join(out)

    except Exception as e:
        return f"Ошибка поиска: {e}"


@app.get("/health")
@app.get("/healthz")
async def health():
    return {
        "status": "ok",
        "uptime": round(time.time() - START_TIME, 2)
    }


@app.get("/v1/models")
async def models_list():

    result = []

    all_models = getattr(
        models,
        "_all_models",
        []
    )

    registry = getattr(
        models,
        "__models__",
        {}
    )

    for model_id in all_models:

        item = {
            "id": model_id,
            "object": "model",
            "owned_by": "g4f"
        }

        try:

            if model_id in registry:

                model_obj, providers = registry[model_id]

                item["name"] = getattr(
                    model_obj,
                    "name",
                    model_id
                )

                item["providers"] = [
                    getattr(
                        p,
                        "__name__",
                        str(p)
                    )
                    for p in providers
                    if getattr(
                        p,
                        "working",
                        True
                    )
                ]

        except Exception:
            pass

        result.append(item)

    return {
        "object": "list",
        "data": result
    }


@app.get("/v1/providers")
async def providers_list():

    try:

        from g4f.Provider import __providers__

        result = []

        for provider in __providers__:

            if getattr(
                provider,
                "working",
                False
            ):

                result.append({
                    "id": provider.__name__,
                    "object": "provider",
                    "url": getattr(
                        provider,
                        "url",
                        None
                    )
                })

        return {
            "object": "list",
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            500,
            f"Provider error: {e}"
        )


@app.get("/v1/search")
async def web_search(q: str):

    if not q.strip():

        raise HTTPException(
            400,
            "Empty query"
        )

    return {
        "query": q,
        "results": search(q)
    }


@app.post("/v1/chat/completions")
async def chat(request: Request):

    if request.stream:

        raise HTTPException(
            400,
            "Streaming is not supported"
        )

    try:

        messages = [
            m.model_dump()
            for m in request.messages
        ]

        if not any(
            m["role"] == "system"
            for m in messages
        ):

            messages.insert(
                0,
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            )

        response = await client.chat.completions.create(
            model=request.model,
            messages=messages,
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

        raise HTTPException(
            500,
            f"G4F error: {e}"
        )


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        access_log=False
    )
