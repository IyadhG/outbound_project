# Detective Module - B2B Lead Generation Pipeline
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p ranking logs output

# Environment variables (to be set at runtime)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for MCP server
EXPOSE 8000

# Default command runs the main pipeline
CMD ["python", "main.py"]
