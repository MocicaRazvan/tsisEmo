FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=5000

ARG OLLAMA_HOST=http://localhost:11434
ARG OLLAMA_MODEL=gemma2

ENV OLLAMA_HOST=${OLLAMA_HOST}
ENV OLLAMA_MODEL=${OLLAMA_MODEL}

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE ${PORT}

CMD ["python", "index.py"]