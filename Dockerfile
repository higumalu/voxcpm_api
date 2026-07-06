FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY voxcpm_api ./voxcpm_api
RUN uv sync --frozen --no-dev

ENV VOXCPM_VOICES_DIR=/app/voices
RUN mkdir -p /app/voices

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "voxcpm_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
