import json

import boto3
import pytest
from moto import mock_aws

import model_registry as reg

BUCKET = "test-model-registry"


@pytest.fixture()
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def test_upload_pushes_all_files_manifest_and_latest(s3, root):
    manifest = reg.upload(BUCKET, "v1", src=root / "model", s3=s3)
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    for name in reg.REQUIRED_FILES:
        assert f"dementia-model/v1/{name}" in keys
    assert "dementia-model/v1/manifest.json" in keys
    latest = json.loads(s3.get_object(Bucket=BUCKET, Key="dementia-model/latest.json")["Body"].read())
    assert latest == {"version": "v1"}
    assert set(manifest["files"]) == set(reg.REQUIRED_FILES)


def test_fetch_roundtrip_latest(s3, root, tmp_path):
    reg.upload(BUCKET, "abc123", src=root / "model", s3=s3)
    reg.fetch(BUCKET, "latest", dest=tmp_path, s3=s3)
    for name in reg.REQUIRED_FILES:
        assert (tmp_path / name).read_bytes() == (root / "model" / name).read_bytes()
    assert (tmp_path / "VERSION").read_text() == "abc123"


def test_fetch_specific_old_version(s3, root, tmp_path):
    reg.upload(BUCKET, "v1", src=root / "model", s3=s3)
    reg.upload(BUCKET, "v2", src=root / "model", s3=s3)
    reg.fetch(BUCKET, "v1", dest=tmp_path, s3=s3)
    assert (tmp_path / "VERSION").read_text() == "v1"


def test_upload_fails_when_model_files_missing(s3, tmp_path):
    with pytest.raises(FileNotFoundError):
        reg.upload(BUCKET, "v1", src=tmp_path, s3=s3)


def test_fetch_detects_corruption(s3, root, tmp_path):
    reg.upload(BUCKET, "v1", src=root / "model", s3=s3)
    s3.put_object(Bucket=BUCKET, Key="dementia-model/v1/scaler.joblib", Body=b"corrupted")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        reg.fetch(BUCKET, "v1", dest=tmp_path, s3=s3)


def test_fetch_unknown_version_fails(s3, tmp_path):
    with pytest.raises(Exception):
        reg.fetch(BUCKET, "does-not-exist", dest=tmp_path, s3=s3)


def test_cli_upload_then_fetch(s3, root, tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    assert reg.main(["upload", "--bucket", BUCKET, "--version", "cli1", "--src", str(root / "model")]) == 0
    assert reg.main(["fetch", "--bucket", BUCKET, "--version", "latest", "--dest", str(tmp_path)]) == 0
    assert (tmp_path / "VERSION").read_text() == "cli1"
