# Lucille render service (Cloud Run: lucille-render). Build from the repo root:
#   docker build -f render_service/Dockerfile .
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY render_service/requirements.txt /app/render_service/requirements.txt
RUN pip install --no-cache-dir -r /app/render_service/requirements.txt
COPY render_service /app/render_service
COPY escape_api/data /app/escape_api/data
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "render_service.app:app", "--host", "0.0.0.0", "--port", "8080"]
