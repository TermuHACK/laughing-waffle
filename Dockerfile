# --- Слой сборки зависимостей (Builder) ---
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Финальный слой (Runner) ---
FROM python:3.11-slim AS runner

WORKDIR /app

# Устанавливаем системные зависимости для работы pydub (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY app.py .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Открываем порт (на Render он динамический, но 8000 ставим как фолбэк для локальных тестов)
EXPOSE 8000

# Обновленный Healthcheck, который использует переменную PORT (или 8000 по умолчанию)
HEALTHCHECK --interval=10s --timeout=3s --start-period=3s --retries=3 \
  CMD python3 -c "import urllib.request, os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8000)}/health')" || exit 1

# Запускаем через uvicorn в shell-формате, чтобы сработала подстановка переменной $PORT от Render
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
