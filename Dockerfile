# syntax=docker/dockerfile:1.7
# Which stage supplies the model:
#   model-registry (default) -> downloaded from the S3 model registry (CD)
#   model-local              -> copied from ./model in the repo (CI smoke test)
ARG MODEL_STAGE=model-registry

# ---------- model from registry ----------
FROM python:3.11-slim AS model-registry
ARG MODEL_REGISTRY_BUCKET
ARG MODEL_VERSION=latest
ARG AWS_DEFAULT_REGION
RUN pip install --no-cache-dir boto3
COPY scripts/model_registry.py /tmp/model_registry.py
# AWS credentials are passed as BuildKit secrets, so they never end up in an image layer.
RUN --mount=type=secret,id=aws_access_key_id \
    --mount=type=secret,id=aws_secret_access_key \
    AWS_ACCESS_KEY_ID="$(cat /run/secrets/aws_access_key_id)" \
    AWS_SECRET_ACCESS_KEY="$(cat /run/secrets/aws_secret_access_key)" \
    python /tmp/model_registry.py fetch \
      --bucket "${MODEL_REGISTRY_BUCKET}" --version "${MODEL_VERSION}" --dest /model

# ---------- model from repo ----------
FROM scratch AS model-local
COPY model/ /model/

FROM ${MODEL_STAGE} AS model-source

# ---------- application ----------
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY templates/ ./templates/
COPY --from=model-source /model/ ./model/

ENV MODEL_DIR=/app/model PYTHONUNBUFFERED=1
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--chdir", "src", "app:app"]
