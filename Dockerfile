FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements (base layer — no torch by default)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    requests \
    gradio \
    PyYAML \
    pytest \
    pytest-asyncio \
    httpx

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose ports for FastAPI backend and Gradio UI
EXPOSE 8000 7860

# Default: run backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
