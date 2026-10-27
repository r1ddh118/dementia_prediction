# syntax=docker/dockerfile:1.7
# Local builds use the model in ./model. CD selects model-registry.
ARG MODEL_STAGE=model-local

FROM python:3.11-slim AS model-registry
ARG MODEL_REGISTRY_BUCKET
ARG MODEL_VERSION=latest
ARG AWS_DEFAULT_REGION
RUN pip install --no-cache-dir boto3
COPY scripts/model_registry.py /tmp/model_registry.py
RUN --mount=type=secret,id=aws_access_key_id \
    --mount=type=secret,id=aws_secret_access_key \
    AWS_ACCESS_KEY_ID="$(cat /run/secrets/aws_access_key_id)" \
    AWS_SECRET_ACCESS_KEY="$(cat /run/secrets/aws_secret_access_key)" \
    AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
    python /tmp/model_registry.py fetch \
      --bucket "${MODEL_REGISTRY_BUCKET}" --version "${MODEL_VERSION}" --dest /model

FROM scratch AS model-local
COPY model/ /model/

FROM ${MODEL_STAGE} AS model-source

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY templates/ ./templates/
COPY --from=model-source /model/ ./model/
ENV MODEL_DIR=/app/model PYTHONUNBUFFERED=1
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health')" || exit 1
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--access-logfile", "-", "--error-logfile", "-", "--bind", "0.0.0.0:5000", "--chdir", "src", "app:app"]
