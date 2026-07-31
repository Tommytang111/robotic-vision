# Clothing Detection API
#   docker build --platform linux/amd64 \
#       --build-arg YOLO_WEIGHTS_URL=https://.../yolov11m-fashionpedia.pt \
#       -t robotic-vision .

FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# CPU index: plain `pip install torch` pulls the ~2.5GB CUDA build.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Weights are baked in at build time so cold starts never wait on a download.
ARG YOLO_WEIGHTS_URL
ARG SAM3_WEIGHTS_URL

RUN mkdir -p /models

RUN if [ -z "$YOLO_WEIGHTS_URL" ]; then \
        echo "ERROR: YOLO_WEIGHTS_URL build arg is required." >&2; \
        exit 1; \
    fi \
    && curl -fSL "$YOLO_WEIGHTS_URL" -o /models/yolov11m-fashionpedia.pt

# Optional: omit to ship a detection-only image.
RUN if [ -n "$SAM3_WEIGHTS_URL" ]; then \
        curl -fSL "$SAM3_WEIGHTS_URL" -o /models/sam3.pt; \
    fi

FROM python:3.11-slim

# libgl1/libglib are required by opencv-python, which ultralytics depends on.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models /models

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_MODEL_PATH=/models/yolov11m-fashionpedia.pt \
    SAM3_MODEL_PATH=/models/sam3.pt \
    PORT=8000

# Redirected to /tmp so the non-root user can write them.
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY service/ /app/

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Shell form so ${PORT} is expanded at runtime.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
