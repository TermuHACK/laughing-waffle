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
