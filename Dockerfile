FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml README.md ./
COPY src/ src/

# Install
RUN pip install --no-cache-dir -e .

# Data volume
RUN mkdir -p /data/.cache
VOLUME /data

# Environment
ENV MEMORY_WORKSPACE=/data
ENV MEMORY_CACHE_DIR=/data/.cache
ENV MEMORY_MODEL=all-MiniLM-L6-v2
ENV MEMORY_HOST=0.0.0.0
ENV MEMORY_PORT=8765
ENV PYTHONPATH=/app/src
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8765

# Default: Streamable HTTP (most compatible with remote clients)
CMD ["python", "-m", "memory_os_ai.server", "--http"]
