FROM docker.io/pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    VIRTUAL_ENV=/opt/venv

ENV PATH=/opt/venv/bin:${PATH}

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN /usr/bin/python -m venv --without-pip --system-site-packages /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip==26.0.1 \
    && /opt/venv/bin/python -m pip install --no-cache-dir --requirement /app/requirements.txt \
    && /opt/venv/bin/python -m pip check

COPY src/nuextract_worker/model_manifest.py /app/src/nuextract_worker/model_manifest.py
COPY download_model.py /app/download_model.py
RUN HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
    python /app/download_model.py \
    && rm -rf /tmp/huggingface

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY src /app/src
COPY handler.py LICENSE NOTICE THIRD_PARTY_NOTICES.md /app/

STOPSIGNAL SIGTERM
CMD ["python", "-u", "handler.py"]
