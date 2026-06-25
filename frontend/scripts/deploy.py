#!/usr/bin/env python3
"""Build the Nuxt site and upload it to S3.

Usage: python scripts/deploy_s3.py --bucket YOUR_BUCKET
"""

from __future__ import annotations

import mimetypes
import os
import subprocess
import sys
from pathlib import Path

import boto3
from decouple import Config, RepositoryIni

CACHE_ASSETS = "public, max-age=31536000, immutable"
CACHE_HTML = "public, max-age=0, must-revalidate"


def content_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def cache_control(path: Path) -> str:
    if path.suffix == ".html":
        return CACHE_HTML
    return CACHE_ASSETS


def main() -> None:
    frontend_dir = Path(__file__).resolve().parent.parent
    repo_root = frontend_dir.parent
    source = frontend_dir / ".output" / "public"

    config = Config(RepositoryIni(str(Path(__file__).resolve().parent / "settings.ini")))
    bucket = config("BUCKET")
    distribution = config("DISTRIBUTION")

    cred = repo_root / ".aws" / "credentials"
    if cred.is_file():
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(cred)
        os.environ["AWS_PROFILE"] = "latentlang"

    os.environ.setdefault("NUXT_PUBLIC_API_BASE", "https://api.latentlang.com")

    subprocess.run(
        ['bash', '-lc', 'source "$NVM_DIR/nvm.sh" && nvm use 20 && npm run generate'],
        cwd=frontend_dir,
        check=True,
    )

    if not source.is_dir():
        sys.exit(f"Missing {source} after build.")

    session = boto3.Session()
    s3 = session.client("s3", region_name="us-east-1")

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objs:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objs})
            for o in objs:
                print(f"deleted {o['Key']}")

    for dirpath, _, filenames in os.walk(source):
        for name in filenames:
            path = Path(dirpath) / name
            key = path.relative_to(source).as_posix()
            s3.upload_file(
                str(path),
                bucket,
                key,
                ExtraArgs={"ContentType": content_type(path), "CacheControl": cache_control(path)},
            )
            print(f"uploaded {key}")

    cf = session.client("cloudfront")
    cf.create_invalidation(
        DistributionId=distribution,
        InvalidationBatch={"Paths": {"Quantity": 1, "Items": ["/*"]}, "CallerReference": str(int(__import__("time").time()))},
    )
    print(f"invalidated {distribution}")


if __name__ == "__main__":
    main()