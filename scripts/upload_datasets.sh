#!/usr/bin/env bash
# One-time script: upload the OASIS dataset xlsx files to S3.
# Run this locally (with valid AWS creds) before your first CI push.
#
# Usage:
#   export DATASETS_BUCKET=your-bucket-name   # e.g. dementia-datasets-yourinitials
#   export AWS_REGION=us-east-1
#   bash scripts/upload_datasets.sh
#
# After running, add DATASETS_BUCKET as a GitHub Actions secret.
set -euo pipefail

BUCKET="${DATASETS_BUCKET:?Set the DATASETS_BUCKET env var first}"
REGION="${AWS_REGION:-us-east-1}"
DATASETS_DIR="$(cd "$(dirname "$0")/../datasets" && pwd)"

echo "Creating bucket s3://${BUCKET} (if it does not exist)..."
if ! aws s3api head-bucket --bucket "${BUCKET}" 2>/dev/null; then
  if [ "${REGION}" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "${BUCKET}"
  else
    aws s3api create-bucket --bucket "${BUCKET}" \
      --create-bucket-configuration LocationConstraint="${REGION}"
  fi
  echo "  Bucket created."
fi

echo "Blocking all public access..."
aws s3api put-public-access-block --bucket "${BUCKET}" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

echo "Uploading datasets/ -> s3://${BUCKET}/datasets/ ..."
aws s3 sync "${DATASETS_DIR}/" "s3://${BUCKET}/datasets/" \
  --exclude "*" --include "*.xlsx" \
  --storage-class STANDARD_IA

echo ""
echo "Done. Files in bucket:"
aws s3 ls "s3://${BUCKET}/datasets/"
echo ""
echo "Next steps:"
echo "  1. Add the following GitHub Actions secret to your repo:"
echo "     DATASETS_BUCKET = ${BUCKET}"
echo "  2. Commit and push .gitignore + workflow changes."
