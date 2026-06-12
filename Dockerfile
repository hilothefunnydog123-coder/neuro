# Container for the Switchboard app (the live project).
FROM python:3.11-slim

WORKDIR /app
COPY switchboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# only the app lives in the image
COPY switchboard/ ./

EXPOSE 8000

# Most hosts inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
