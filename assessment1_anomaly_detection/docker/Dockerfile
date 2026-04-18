# Use slim Python image to keep the container lightweight
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-trained models
COPY src/ ./src/
COPY models/ ./models/

# Set environment variable so app.py knows where to find models
ENV MODEL_DIR=/app/models

EXPOSE 8000

# Run the FastAPI server with uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
