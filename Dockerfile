# PLOMID ODC runtime image (Phase 12) — Python-only, offline-capable.
# Weights are NEVER baked in (fetch at deploy, verify by sha256).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY odc/ ./odc/
COPY scripts/fetch_models.py scripts/export_onnx.py ./scripts/
COPY models/manifests/ ./models/manifests/
COPY configs/ ./configs/
COPY sql/ ./sql/
COPY eval/datasets/ eval/annotations/ ./eval/
# Pinned versions mirror requirements.lock (mac wheels resolve to linux here).
RUN pip install --upgrade pip && pip install \
    "pydantic>=2.9,<3" "pydantic-settings>=2.5,<3" "pyyaml>=6,<7" \
    "psycopg[binary]>=3.2" "numpy>=2,<3" "pillow>=12" \
    "opencv-python-headless>=4.10" "onnxruntime>=1.20" \
    "fastapi>=0.115" "uvicorn[standard]>=0.30" "jinja2>=3.1" "mediapipe>=1.0" \
    "torch==2.14.1" "torchvision==0.29.1" "timm==1.0.30" \
    "transformers==5.19.0" "safetensors>=0.5" \
    && rm -rf /root/.cache

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -m odc.cli health || exit 1
ENTRYPOINT ["python", "-m", "odc.cli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
