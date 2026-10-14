# Churn prediction API. Build the model folder first: `make package` (writes build/model).
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/Hizafa-Nadeem/TelecomChurnPrediction" \
      org.opencontainers.image.description="Telecom churn prediction API: calibrated CatBoost with per-customer reasons" \
      org.opencontainers.image.licenses="MIT"

# CatBoost needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app

WORKDIR /app
COPY requirements-api.txt ./
COPY build/model ./model

# The model folder pins the exact packages it was trained with. The service only needs to load and
# run it, so the light mlflow-skinny replaces the full mlflow pin.
RUN grep -v '^mlflow==' model/requirements.txt > /tmp/model-requirements.txt \
    && pip install --no-cache-dir --retries 10 --default-timeout 60 \
        -r requirements-api.txt -r /tmp/model-requirements.txt mlflow-skinny==3.16.1

COPY api ./api

ENV MODEL_URI=/app/model \
    WEB_CONCURRENCY=4
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
