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

# NOTE: Do NOT bundle Firebase service account JSON in the container image.
# In production (Cloud Run), Firebase Admin SDK automatically uses Application
# Default Credentials from the GCP metadata server — no JSON file needed.
# The JSON file is only for local development.

# Upgrade pip and install all dependencies
RUN pip install --no-cache-dir --upgrade pip --root-user-action=ignore \
    && pip install --no-cache-dir -r requirements.txt

# Verify no dependency conflicts
RUN pip check || echo "Warning: Some dependencies may have conflicts."

# Non-root user (CIS benchmark for containers)
RUN useradd -m -u 1000 lucille && chown -R lucille:lucille /app
USER lucille

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
