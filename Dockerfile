# SmartShop rescan decision API. The model bundle is NOT baked in: it is mounted at
# /app/artifacts, so the weekly bundle ships without rebuilding the image, and the API
# hot-reloads it.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SMARTSHOP_BUNDLE=/app/artifacts

WORKDIR /app
COPY docker/requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 app && mkdir -p /app/artifacts && chown app /app/artifacts
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "smartshop.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
