FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only torch (much smaller than the default CUDA build)
RUN if [ "$(uname -m)" = "x86_64" ]; then \
      pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu; \
    else \
      pip install torch==2.1.2; \
    fi

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake models into the image so the container starts fast and works offline
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

RUN useradd -m -u 1000 user && mkdir -p /app/storage && chown -R user /app /opt/hf-cache
USER user

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:10000/')" || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
