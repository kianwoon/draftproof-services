"""R2 storage — upload report files, return presigned URLs."""

import json
import logging
import tempfile
from typing import Dict

import boto3
from botocore.config import Config as BotoConfig

from .config import settings

logger = logging.getLogger("storage")


def _client():
    return boto3.client(
        "s3",
        endpoint_url=settings.R2_ENDPOINT_URL,
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        config=BotoConfig(
            signature_version="s3v4",
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    )


def upload_report_files(
    job_id: str,
    md_text: str,
    pdf_bytes: bytes,
    result_dict: dict,
    paragraph_explanations: dict | None = None,
) -> Dict[str, str]:
    """Upload MD, PDF, JSON, and optional explanation sidecar to R2."""
    s3 = _client()
    bucket = settings.R2_BUCKET_NAME
    prefix = f"reports/{job_id}"
    urls = {}

    uploads = [
        ("report.json", json.dumps(result_dict, indent=2, ensure_ascii=False).encode(), "application/json", "json"),
        ("report.md", md_text.encode(), "text/markdown", "md"),
        ("report.pdf", pdf_bytes, "application/pdf", "pdf"),
    ]
    if paragraph_explanations is not None:
        uploads.append((
            "paragraph_explanations.json",
            json.dumps(paragraph_explanations, indent=2, ensure_ascii=False).encode(),
            "application/json",
            "paragraph_explanations",
        ))

    for filename, data, content_type, url_key in uploads:
        key = f"{prefix}/{filename}"
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        urls[url_key] = _presign(s3, bucket, key)

    return urls


def _presign(s3, bucket: str, key: str, expires: int = 3600) -> str:
    """Generate a presigned URL valid for `expires` seconds."""
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def upload_rewrite_files(
    scan_id: str,
    md_text: str,
    pdf_bytes: bytes,
    json_data: dict,
    rewritten_text: str,
    debug_log: str = "",
) -> Dict[str, str]:
    """Upload rewrite results to R2 under reports/{scan_id}/rewrite/."""
    s3 = _client()
    bucket = settings.R2_BUCKET_NAME
    prefix = f"reports/{scan_id}/rewrite"
    urls = {}

    uploads = [
        ("rewrite.json", json.dumps(json_data, indent=2, ensure_ascii=False).encode(), "application/json"),
        ("rewrite.md", md_text.encode(), "text/markdown"),
        ("rewrite.pdf", pdf_bytes, "application/pdf"),
        ("rewritten.txt", rewritten_text.encode("utf-8"), "text/plain"),
        ("rewrite.log", debug_log.encode("utf-8"), "text/plain"),
    ]
    for filename, data, content_type in uploads:
        if not data:
            logger.warning("Skipping upload of %s — empty data", filename)
            continue
        key = f"{prefix}/{filename}"
        logger.info("Uploading %s (%d bytes)", filename, len(data))
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        urls[filename] = _presign(s3, bucket, key)

    return urls


def upload_rewrite_checkpoint(
    scan_id: str,
    checkpoint_json: dict,
    rewritten_text: str,
    md_text: str = "",
) -> Dict[str, str]:
    """Persist the latest accepted rewrite checkpoint before the full pipeline ends."""
    s3 = _client()
    bucket = settings.R2_BUCKET_NAME
    prefix = f"reports/{scan_id}/rewrite"
    urls = {}
    json_bytes = json.dumps(checkpoint_json, indent=2, ensure_ascii=False).encode()
    text_bytes = rewritten_text.encode("utf-8")
    uploads = [
        ("checkpoint.json", json_bytes, "application/json"),
        ("checkpoint_rewritten.txt", text_bytes, "text/plain"),
        # Keep the normal report keys warm as a recoverable partial rewrite.
        # The final pipeline upload overwrites these when it completes.
        ("rewrite.json", json_bytes, "application/json"),
        ("rewritten.txt", text_bytes, "text/plain"),
    ]
    if md_text:
        uploads.extend([
            ("checkpoint.md", md_text.encode("utf-8"), "text/markdown"),
            ("rewrite.md", md_text.encode("utf-8"), "text/markdown"),
        ])

    for filename, data, content_type in uploads:
        if not data:
            logger.warning("Skipping checkpoint upload of %s — empty data", filename)
            continue
        key = f"{prefix}/{filename}"
        logger.info("Uploading rewrite checkpoint %s (%d bytes)", filename, len(data))
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        urls[filename] = _presign(s3, bucket, key)
    return urls
