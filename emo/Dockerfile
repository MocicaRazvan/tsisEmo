FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=5000

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirement.txt .

RUN pip install --upgrade pip
RUN pip install -r requirement.txt --extra-index-url https://download.pytorch.org/whl/cu121

COPY src /app/src
COPY data /app/data
COPY config /app/config

EXPOSE ${PORT}

# CMD ["python", "src/Prediction.py"]
# CMD ["tail", "-f", "/dev/null"]
CMD [ "python","src/index.py" ]
