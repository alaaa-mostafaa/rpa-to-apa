FROM python:3.12-slim

WORKDIR /app

# Install git and other necessary tools for the pipeline
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set default environment variables
ENV PYTHONUNBUFFERED=1
ENV WEBHOOK_PORT=8090
ENV CI_AGENT_MODEL=deepseek-chat
ENV LLM_PROVIDER=deepseek

# Expose the webhook/dashboard port
EXPOSE 8090

# Start the FastAPI server
CMD ["python", "src/web/server.py"]
