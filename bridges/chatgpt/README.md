# ChatGPT / OpenAI Remote Bridge — Memory OS AI

## Architecture

ChatGPT (web) only supports **remote MCP servers** (SSE or Streamable HTTP).
You need to run Memory OS AI as a server accessible over HTTPS.

## Quick Start (local testing)

```bash
# Run SSE transport on localhost:8765
python -m memory_os_ai.server --sse

# With API key authentication
MEMORY_API_KEY=your-secret-key python -m memory_os_ai.server --sse

# Or Streamable HTTP (newer, preferred)
python -m memory_os_ai.server --http
```

## Production Deployment

### Option 1: ngrok tunnel (quick)
```bash
# Terminal 1: start the server
MEMORY_API_KEY=your-secret python -m memory_os_ai.server --sse

# Terminal 2: expose via ngrok
ngrok http 8765
# → https://abc123.ngrok.io
```

### Option 2: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
ENV MEMORY_WORKSPACE=/data
ENV MEMORY_CACHE_DIR=/data/.cache
ENV MEMORY_API_KEY=change-me
EXPOSE 8765
CMD ["python", "-m", "memory_os_ai.server", "--http"]
```

### Option 3: Cloud Run / Railway / Fly.io
```bash
# Example with fly.io
fly deploy --env MEMORY_API_KEY=your-secret
```

## ChatGPT Configuration

1. Go to **ChatGPT Settings → Connections → Add MCP Server**
2. Enter your server URL: `https://your-server.com/sse` (SSE) or `https://your-server.com/mcp/` (HTTP)
3. Add authentication header: `Authorization: Bearer your-secret-key`
4. Memory OS AI tools appear in deep research and MCP apps

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_HOST` | `0.0.0.0` | Bind address |
| `MEMORY_PORT` | `8765` | Bind port |
| `MEMORY_API_KEY` | *(none)* | API key for auth (recommended in production) |
| `MEMORY_WORKSPACE` | `.` | Root workspace directory |
| `MEMORY_CACHE_DIR` | `.` | Cache + index directory |
| `MEMORY_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model |
