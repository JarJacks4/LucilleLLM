# Use an official Python base image
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    libdbus-1-dev \
    pkg-config \
    gobject-introspection \
    libgirepository1.0-dev \
    gir1.2-glib-2.0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files to the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
