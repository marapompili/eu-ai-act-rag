FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/opt/hf

# 1. CPU-only PyTorch first: the default wheel pulls ~4 GB of CUDA libraries
#    that are useless in a container without an NVIDIA GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. The other libraries: cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Embedding model baked into the image, so startup needs no network.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
ENV HF_HUB_OFFLINE=1

# 4. Code last: it changes most often.
COPY src/ src/

RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
