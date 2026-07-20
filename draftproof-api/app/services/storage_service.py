import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from fastapi import UploadFile
from app.config import UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE, UPLOAD_RETENTION_HOURS
from app.models import DocumentOut

_log = logging.getLogger("storage_service")

# Opportunistic sweep of stale UPLOAD_DIR files (backstop, not the primary cleanup —
# see scan_service._delete_upload_best_effort for the delete-on-extract path). Throttled
# in-process so a burst of uploads doesn't turn every save_upload/save_text call into a
# full directory listdir+stat; a small race (two sweeps close together) is harmless.
_SWEEP_MIN_INTERVAL_SECONDS = 3600
_last_sweep_monotonic = 0.0


def _write_file_sync(dest: str, content: bytes):
    with open(dest, "wb") as f:
        f.write(content)


def _write_text_sync(dest: str, text: str):
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)


def _decodes_as_utf8(chunk: bytes, *, may_be_truncated: bool) -> bool:
    """True if `chunk` is valid UTF-8.

    `may_be_truncated` tolerates a multi-byte sequence cut at the end of the chunk, but
    ONLY when the chunk is exactly a full read (i.e. more file data is still coming, per
    the caller-supplied read size) -- otherwise this chunk IS the entire (short) file, an
    invalid trailing byte is a real decode error, and forgiving it would let e.g. a 4-byte
    latin-1 file through unchecked.

    Only strict UTF-8 is accepted (no latin-1 fallback): scan_service._read_document_text_sync
    reads uploaded .txt files with open(path, encoding="utf-8") and no fallback, so a file
    that fails strict UTF-8 here would fail identically downstream with an unhandled
    UnicodeDecodeError -- rejecting it here just moves that failure earlier, to the endpoint
    that already gives a validation-error response instead of a 500.
    """
    try:
        chunk.decode("utf-8")
        return True
    except UnicodeDecodeError as e:
        if may_be_truncated and e.end == len(chunk) and (len(chunk) - e.start) <= 3:
            try:
                chunk[: e.start].decode("utf-8")
                return True
            except UnicodeDecodeError:
                return False
        return False


def _sniff_first_chunk(ext: str, chunk: bytes, *, may_be_truncated: bool) -> None:
    """Validate the first chunk's content matches the claimed extension, rejecting with
    the same error style as the extension check in save_upload (a plain ValueError).

    Only .txt reaches here today (ALLOWED_EXTENSIONS) — .txt has no magic bytes, so it
    is validated by decodability. If a binary type is ever re-added to
    ALLOWED_EXTENSIONS, it needs BOTH a magic-byte branch here and a real server-side
    extractor in scan_service (see the ALLOWED_EXTENSIONS comment in app/config.py).
    """
    if ext == ".txt":
        if b"\x00" in chunk:
            raise ValueError("File content does not match a text file")
        if not _decodes_as_utf8(chunk, may_be_truncated=may_be_truncated):
            raise ValueError("File content does not match a text file")


def _sweep_stale_uploads_sync(max_age_hours: int) -> None:
    cutoff = time.time() - max_age_hours * 3600
    try:
        names = os.listdir(UPLOAD_DIR)
    except OSError:
        return
    for name in names:
        path = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError as e:
            _log.warning("Stale-upload sweep failed for %s: %s", path, e)


async def _maybe_sweep_stale_uploads() -> None:
    """Opportunistic, throttled sweep of UPLOAD_DIR — invoked from save_upload/save_text
    instead of a dedicated scheduler/daemon. Backstop for files that are never consumed
    (e.g. an upload whose scan is never created), since the delete-on-extract path in
    scan_service handles the normal single-read lifecycle."""
    global _last_sweep_monotonic
    now = time.monotonic()
    if now - _last_sweep_monotonic < _SWEEP_MIN_INTERVAL_SECONDS:
        return
    _last_sweep_monotonic = now
    await asyncio.to_thread(_sweep_stale_uploads_sync, UPLOAD_RETENTION_HOURS)


async def save_upload(file: UploadFile) -> DocumentOut:
    # Sanitize filename — strip any path components
    safe_filename = os.path.basename(file.filename or "upload.txt")
    ext = os.path.splitext(safe_filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type {ext} not allowed")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    await _maybe_sweep_stale_uploads()
    doc_id = str(uuid.uuid4())
    dest = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")

    # Read in 1 MiB chunks and abort as soon as the limit is exceeded, so a grossly
    # oversized upload cannot force us to buffer the entire body before the size check
    # (L12). Bounds memory/temp pressure to ~MAX_FILE_SIZE + one chunk.
    read_size = 1024 * 1024
    buf = bytearray()
    is_first_chunk = True
    while True:
        chunk = await file.read(read_size)
        if not chunk:
            break
        if is_first_chunk:
            # Sniff from the chunk already in hand -- no extra buffering/reads just to
            # validate content type. A full-size chunk means more file data may still be
            # coming (relevant to the UTF-8 truncation tolerance below); a short chunk
            # means this already-read data IS the whole file.
            _sniff_first_chunk(ext, chunk, may_be_truncated=len(chunk) == read_size)
            is_first_chunk = False
        buf.extend(chunk)
        if len(buf) > MAX_FILE_SIZE:
            raise ValueError(f"File exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
    await asyncio.to_thread(_write_file_sync, dest, bytes(buf))

    return DocumentOut(id=doc_id, filename=safe_filename, created_at=datetime.now(timezone.utc))


async def save_text(text: str) -> DocumentOut:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    await _maybe_sweep_stale_uploads()
    doc_id = str(uuid.uuid4())
    dest = os.path.join(UPLOAD_DIR, f"{doc_id}.txt")
    await asyncio.to_thread(_write_text_sync, dest, text)
    return DocumentOut(id=doc_id, filename=f"{doc_id}.txt", created_at=datetime.now(timezone.utc))
