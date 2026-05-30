FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

COPY api /app/api
COPY frontend /app/frontend
COPY tech-doc-rag/src /app/tech-doc-rag/src
COPY tech-doc-rag/data /app/tech-doc-rag/data
COPY tech-doc-rag/.faiss_index /app/tech-doc-rag/.faiss_index

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
