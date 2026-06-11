# Container for the YN Neuro forecaster API.
# Deploy to Render / Railway / Fly.io / Hugging Face Spaces, then point the
# Lovable frontend at the resulting public URL.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_CACHE_DIR=/app/checkpoints
EXPOSE 8000

# Most hosts inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
