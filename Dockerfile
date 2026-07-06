FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download ChromaDB embedding model so first query doesn't time out
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()"

# Seed faqs.csv into ChromaDB at build time — DB is ready before app starts
RUN python ingest.py

EXPOSE 5001

# Single worker: ChromaDB PersistentClient caches the vector index per process,
# so docs added via /api/teach in one worker are invisible to queries in another.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "4", "--timeout", "120"]
