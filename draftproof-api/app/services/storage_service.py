import os
import uuid
from datetime import datetime
from fastapi import UploadFile
from app.config import UPLOAD_DIR, ALLOWED_EXTENSIONS
from app.models import DocumentOut


async def save_upload(file: UploadFile) -> DocumentOut:
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type {ext} not allowed")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    doc_id = str(uuid.uuid4())
    dest = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")

    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    return DocumentOut(id=doc_id, filename=file.filename, created_at=datetime.now())


async def save_text(text: str) -> DocumentOut:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    doc_id = str(uuid.uuid4())
    dest = os.path.join(UPLOAD_DIR, f"{doc_id}.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    return DocumentOut(id=doc_id, filename=f"{doc_id}.txt", created_at=datetime.now())
