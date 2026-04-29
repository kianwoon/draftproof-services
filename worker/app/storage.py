"""R2 storage — upload report files, return presigned URLs."""

import json
import tempfile
from typing import Dict

import boto3
from botocore.config import Config as BotoConfig

from .config import settings


def _client():
    return boto3.client(
        "s3",
        endpoint_url=settings.R2_ENDPOINT_URL,
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        config=BotoConfig(signature_version="s3v4"),
    )


def upload_report_files(job_id: str, md_text: str, pdf_bytes: bytes, result_dict: dict) -> Dict[str, str]:
    """Upload MD, PDF, and JSON reports to R2. Returns {type: url}."""
    s3 = _client()
    bucket = settings.R2_BUCKET_NAME
    prefix = f"reports/{job_id}"
    urls = {}

    for ext, data in [
        ("json", json.dumps(result_dict, indent=2, ensure_ascii=False).encode()),
        ("md", md_text.encode()),
        ("pdf", pdf_bytes),
    ]:
        key = f"{prefix}/report.{ext}"
        content_type = {
            "json": "application/json",
            "md": "text/markdown",
            "pdf": "application/pdf",
        }[ext]
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        urls[ext] = _presign(s3, bucket, key)

    return urls


def _presign(s3, bucket: str, key: str, expires: int = 3600) -> str:
    """Generate a presigned URL valid for `expires` seconds."""
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )
