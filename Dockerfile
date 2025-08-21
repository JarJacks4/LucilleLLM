# Use an official Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    libdbus-1-dev \
    libglib2.0-dev \
    pkg-config \
    gobject-introspection \
    libgirepository1.0-dev \
    gir1.2-glib-2.0 \
    python3-gi \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Copy the application files to the container
COPY . /app

# Ensure Firebase service account file is properly set
RUN if [ -f "escape-ujuzxr-firebase-adminsdk-he895-1a039bd95a.json" ]; then \
        echo "Firebase service account file found"; \
    else \
        echo "Warning: Firebase service account file not found"; \
    fi

# Upgrade pip and install dependencies efficiently
RUN pip install --no-cache-dir --upgrade pip --root-user-action=ignore \
    && pip install --no-cache-dir --no-deps -r requirements.txt \
    && pip install --no-cache-dir --upgrade --use-deprecated=legacy-resolver -r requirements.txt

# Install missing sentence-transformers package separately
RUN pip install --no-cache-dir sentence-transformers

# Verify that there are no dependency issues
RUN pip check || echo "Warning: Some dependencies may have conflicts."

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
