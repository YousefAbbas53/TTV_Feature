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
ENV HF_HOME=/app/hf_cache
ENV MISTRAL_ADAPTER_DIR=/app/models/mistral-finetuned
ENV MISTRAL_ENABLED=0
ENV VIDEO_MODEL_PRESET=wan_t2v_1_3b

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
RUN awk '!/^rembg(\[[^]]+\])?==/ { print }' /app/Wan2GP/requirements.txt > /tmp/wan2gp_requirements_compat.txt \
    && python -m pip install -r /tmp/wan2gp_requirements_compat.txt

# Force upgrade PyTorch and CUDA dependencies to support Blackwell GPUs (SM_100)
RUN pip install --no-cache-dir --upgrade \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

COPY . /app

COPY wgp_headless_t2v_fix.py /app/Wan2GP/wgp_headless_t2v_fix.py

RUN mkdir -p /app/outputs /app/tmp /app/hf_cache

# NOTE: Model weights are downloaded on first cold start (not during build)
# because GitHub Actions build servers have no GPU, causing preload_model.py to fail.
# RunPod workers have GPU access at runtime, so the download will succeed there.

CMD ["python3", "-u", "handler.py"]
