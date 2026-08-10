import os
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import uvicorn

from g4f.client import AsyncClient
from duckduckgo_search import DDGS

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

g4f_client = AsyncClient()
START_TIME = time.time()

API_KEY = os.getenv("AGENT_API_KEY", "")


SYSTEM_PROMPT = r"""
Ты — автономный Linux-агент и терминальный помощник пользователя.

Ты можешь самостоятельно использовать инструменты локального компьютера.

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:

1. Выполнение Bash-команды:

<<<TOOL bash>>>
команда
<<<END_TOOL>>>

2. Чтение файла:

<<<TOOL read_file>>>
/путь/к/файлу
<<<END_TOOL>>>

3. Запись файла:

<<<TOOL write_file>>>
/путь/к/файлу
содержимое файла
<<<END_TOOL>>>

4. Поиск по файлам через grep:

<<<TOOL grep>>>
параметры grep
<<<END_TOOL>>>

Примеры:

<<<TOOL grep>>>
-rni "TODO" /home/user/project
<<<END_TOOL>>>

<<<TOOL grep>>>
-r "error" /var/log
<<<END_TOOL>>>

5. Интернет-поиск:

<<<TOOL web_search>>>
поисковый запрос
<<<END_TOOL>>>


ПРАВИЛА АГЕНТА:

- Ты настоящий автономный агент.
- Не ограничивайся одним ответом, если задачу можно решить действиями.
- После каждого инструмента анализируй полный результат.
- При необходимости вызывай следующий инструмент.
- Можно выполнять много последовательных инструментов.
- Не утверждай, что действие выполнено, пока не получил результат.
- Не выдумывай stdout, stderr или содержимое файлов.
- Для диагностики системы используй bash.
- Для поиска конкретного текста в проекте используй grep.
- Для чтения небольшого файла используй read_file.
- Для создания или изменения файла используй write_file.
- После изменения файла желательно проверить его содержимое или запустить соответствующую проверку.
- Если команда завершилась ошибкой, проанализируй ошибку и попробуй исправить её.
- Для актуальной информации используй web_search.
- Не используй web_search без необходимости.
- Не удаляй данные без необходимости.
- Перед потенциально разрушительными операциями учитывай последствия.
- После выполнения задачи сообщи краткий итог.

РАБОЧИЙ ЦИКЛ:

1. Понять задачу.
2. Исследовать состояние системы.
3. Найти необходимые файлы.
4. Выполнить изменения.
5. Проверить изменения.
6. Исправить обнаруженные проблемы.
7. Завершить задачу.

Инструменты выполняются НЕ на сервере с LLM.
Они выполняются локальным клиентом пользователя.

Если инструмент не нужен — просто ответь пользователю обычным текстом.
"""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
    web_search: Optional[bool] = False


def check_auth(authorization: Optional[str]):
    if not API_KEY:
        return

    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


def search_ddg(
    query: str,
    max_results: int = 5
) -> str:

    try:
        results = list(
            DDGS().text(
                query,
                max_results=max_results
            )
        )

        if not results:
            return "Результаты поиска не найдены."

        output = []

        for i, result in enumerate(results, 1):
            output.append(
                f"[{i}] {result.get('title', '')}\n"
                f"URL: {result.get('href', '')}\n"
                f"{result.get('body', '')}"
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Ошибка DuckDuckGo: {e}"


def get_g4f_models():

    from g4f import models

    result = []

    all_models = getattr(
        models,
        "_all_models",
        []
    )

    model_registry = getattr(
        models,
        "__models__",
        {}
    )

    for model_id in all_models:

        item = {
            "id": model_id,
            "object": "model",
            "owned_by": "g4f",
        }

        try:
            if model_id in model_registry:

                model_obj, providers = model_registry[
                    model_id
                ]

                item["name"] = getattr(
                    model_obj,
                    "name",
                    model_id
                )

                item["providers"] = []

                for provider in providers:

                    if not getattr(
                        provider,
                        "working",
                        True
                    ):
                        continue

                    item["providers"].append(
                        getattr(
                            provider,
                            "__name__",
                            str(provider)
                        )
                    )

        except Exception:
            pass

        result.append(item)

    return result


@app.get("/health")
@app.get("/healthz")
async def health():

    return {
        "status": "ok",
        "uptime_seconds": round(
            time.time() - START_TIME,
            2
        )
    }


@app.get("/v1/models")
async def list_models(
    authorization: Optional[str] = Header(None)
):

    check_auth(authorization)

    try:

        return {
            "object": "list",
            "data": get_g4f_models()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"G4F model discovery error: {e}"
        )


@app.get("/v1/providers")
async def list_providers(
    authorization: Optional[str] = Header(None)
):

    check_auth(authorization)

    try:

        from g4f.Provider import __providers__

        providers = []

        for provider in __providers__:

            if not getattr(
                provider,
                "working",
                False
            ):
                continue

            providers.append({
                "id": provider.__name__,
                "object": "provider",
                "label": getattr(
                    provider,
                    "label",
                    provider.__name__
                ),
                "url": getattr(
                    provider,
                    "url",
                    None
                )
            })

        return {
            "object": "list",
            "data": providers
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"G4F provider discovery error: {e}"
        )


@app.get("/v1/search")
async def search(
    q: str,
    authorization: Optional[str] = Header(None)
):

    check_auth(authorization)

    if not q.strip():

        raise HTTPException(
            status_code=400,
            detail="Empty query"
        )

    return {
        "query": q,
        "results": search_ddg(q)
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):

    check_auth(authorization)

    if request.stream:

        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported"
        )

    try:

        raw_messages = [
            message.model_dump()
            for message in request.messages
        ]

        if not any(
            message.get("role") == "system"
            for message in raw_messages
        ):

            raw_messages.insert(
                0,
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            )

        if request.web_search:

            last_user = ""

            for message in reversed(raw_messages):

                if message.get("role") == "user":

                    last_user = message.get(
                        "content",
                        ""
                    )

                    break

            if last_user:

                results = search_ddg(last_user)

                raw_messages.append({
                    "role": "system",
                    "content":
                        "Результаты интернет-поиска:\n\n"
                        + results
                })

        response = await g4f_client.chat.completions.create(
            model=request.model,
            messages=raw_messages,
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
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"G4F Execution Error: {e}"
        )


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.getenv("PORT", "8000")
        ),
        access_log=False
    )
