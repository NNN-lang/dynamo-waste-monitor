# ─────────────────────────────────────────────────────────────────────────────
# Dynamo GPU Waste Monitor — Docker Image
# ─────────────────────────────────────────────────────────────────────────────
# Build:
#   docker build -t dynamo-waste-monitor .
#
# Run (simulation mode — no GPU needed):
#   docker run dynamo-waste-monitor --simulate --once
#
# Run (real GPU mode):
#   docker run --gpus all dynamo-waste-monitor --once
#
# Run (connect to Dynamo 1.0):
#   docker run --gpus all --network host \
#     dynamo-waste-monitor \
#     --dynamo-endpoint http://localhost:8080 --dashboard
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="github.com/NNN-lang/dynamo-waste-monitor"
LABEL description="Real-time idle GPU detection for NVIDIA Dynamo 1.0"
LABEL version="0.1.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Try to install nvidia-ml-py (optional — fails gracefully without GPU)
RUN pip install --no-cache-dir nvidia-ml-py || echo "nvidia-ml-py not available — simulation mode only"

# Copy source
COPY dynamo_waste_monitor.py .
COPY pyproject.toml .

# Health check — runs monitor once in simulation mode
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python dynamo_waste_monitor.py --simulate --once || exit 1

# Default: simulation mode dashboard
ENTRYPOINT ["python", "dynamo_waste_monitor.py"]
CMD ["--simulate", "--once"]
