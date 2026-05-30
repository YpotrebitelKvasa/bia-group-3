FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt1-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и конфиги
COPY src/ ./src/
COPY configs/ ./configs/

# Копируем только небольшие обработанные данные (индексы монтируем или скачиваем)
COPY data/processed/ ./data/processed/

# Создаём директорию для индексов (монтируется при запуске)
RUN mkdir -p ./data/indices

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]