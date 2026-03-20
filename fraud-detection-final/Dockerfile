# =============================================================
# Dockerfile — Multi-stage build for the Fraud Detection API
# =============================================================
# Build:  docker build -t fraud-detection-api .
# Run:    docker run -p 8000:8000 fraud-detection-api
#
# Improvements vs original:
#   - Multi-stage build: deps layer + slim runtime (smaller image)
#   - Non-root user (appuser) — security best practice
#   - HEALTHCHECK instruction — Docker/k8s can detect unhealthy state
#   - Layer caching optimized: requirements installed before code copy
# =============================================================

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/packages -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /install/packages /usr/local

# Copy application code
COPY api/      ./api/
COPY model/    ./model/

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check — tests the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
