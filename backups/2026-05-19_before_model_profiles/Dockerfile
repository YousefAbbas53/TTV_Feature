FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV APP_HOME=/app
ENV WAN_REPO_DIR=/app/Wan2GP
ENV WAN_HEADLESS_SCRIPT=/app/Wan2GP/wgp_headless_t2v_fix.py
ENV WAN_TEMPLATE_PATH=/app/Wan2GP/defaults/t2v.json
ENV VIDEO_OUTPUT_DIR=/app/outputs
ENV VIDEO_TEMP_DIR=/app/tmp
ENV MISTRAL_ADAPTER_DIR=/app/models/mistral-finetuned
ENV MISTRAL_ENABLED=0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install -r /app/requirements.txt

RUN git clone https://github.com/deepbeepmeep/Wan2GP.git /app/Wan2GP
RUN python -m pip install -r /app/Wan2GP/requirements.txt

COPY . /app

COPY wgp_headless_t2v_fix.py /app/Wan2GP/wgp_headless_t2v_fix.py

RUN mkdir -p /app/outputs /app/tmp

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
