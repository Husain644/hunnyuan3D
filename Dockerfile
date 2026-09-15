# Hunyuan3D 2.1 API on T4 (14.6 GB usable). CUDA 12.x base.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-venv build-essential cmake \
    libgl1 libglib2.0-0 libgomp1 libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Python deps
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# 2) Hunyuan3D-2.1 source + C++ custom rasterizer (must compile on host arch).
ARG HY3D_REPO=https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
RUN git clone --depth 1 ${HY3D_REPO} /opt/Hunyuan3D-2.1 && \
    cd /opt/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && \
    python3 setup.py install

# 3) Service itself
COPY app/ ./app/
COPY run.sh .

ENV PYTHONPATH=/app:/opt/Hunyuan3D-2.1:$PYTHONPATH \
    HY3D_OUTPUT_DIR=/data/glb \
    HY3D_JOB_DIR=/data/jobs \
    HF_HOME=/app/.cache/hf

VOLUME ["/data"]
EXPOSE 8080

ENTRYPOINT ["bash", "./run.sh"]