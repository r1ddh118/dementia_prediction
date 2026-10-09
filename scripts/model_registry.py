"""Tiny S3-backed model registry.

Layout inside the (versioned) bucket:

    <prefix>/<version>/<file>.joblib   immutable copy of every release
    <prefix>/<version>/manifest.json   file list + sha256 checksums
    <prefix>/latest.json               pointer to the newest version

Usage:
    python scripts/model_registry.py upload --bucket B --version <git-sha> [--src model]
    python scripts/model_registry.py fetch  --bucket B --version latest    [--dest model]

Credentials/region come from the standard AWS env vars
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION / AWS_REGION).
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

REQUIRED_FILES = (
    "dementia_model.joblib",
    "scaler.joblib",
    "label_encoder.joblib",
    "selector.joblib",
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def upload(bucket, version, src="model", prefix="dementia-model", s3=None):
    s3 = s3 or boto3.client("s3")
    src = Path(src)
    missing = [f for f in REQUIRED_FILES if not (src / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model files in {src}: {missing}")

    files = {}
    for name in REQUIRED_FILES:
        path = src / name
        s3.upload_file(str(path), bucket, f"{prefix}/{version}/{name}")
        files[name] = _sha256(path)

    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    s3.put_object(Bucket=bucket, Key=f"{prefix}/{version}/manifest.json",
                  Body=json.dumps(manifest, indent=2).encode())
    s3.put_object(Bucket=bucket, Key=f"{prefix}/latest.json",
                  Body=json.dumps({"version": version}).encode())
    return manifest


def fetch(bucket, version="latest", dest="model", prefix="dementia-model", s3=None):
    s3 = s3 or boto3.client("s3")
    if version == "latest":
        obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/latest.json")
        version = json.loads(obj["Body"].read())["version"]

    obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/{version}/manifest.json")
    manifest = json.loads(obj["Body"].read())

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["files"].items():
        target = dest / name
        s3.download_file(bucket, f"{prefix}/{version}/{name}", str(target))
        if _sha256(target) != expected:
            raise RuntimeError(f"Checksum mismatch for {name} (version {version})")
    (dest / "VERSION").write_text(version)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("upload", "fetch"):
        p = sub.add_parser(name)
        p.add_argument("--bucket", required=True)
        p.add_argument("--version", required=True)
        p.add_argument("--prefix", default="dementia-model")
        p.add_argument("--src" if name == "upload" else "--dest",
                       default="model", dest="path")
    args = parser.parse_args(argv)

    if args.cmd == "upload":
        m = upload(args.bucket, args.version, args.path, args.prefix)
        print(f"Uploaded model version {m['version']} to s3://{args.bucket}/{args.prefix}/")
    else:
        m = fetch(args.bucket, args.version, args.path, args.prefix)
        print(f"Fetched model version {m['version']} into {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
