"""Upload validation tests for storage_service.save_upload.

ALLOWED_EXTENSIONS is narrowed to .txt (app/config.py): no PDF/DOCX extractor exists
anywhere in the API or worker, so those uploads always crashed scan creation with a
masked 500 — they are rejected up front now, at the endpoint, with a clear error.
These tests pin BOTH gates: the extension gate (rejects .pdf/.docx/anything else,
even with genuine magic bytes) and the .txt content sniff (rejects binary/NUL/non-UTF-8
bytes hiding behind a .txt name).
"""

import io

import pytest
from fastapi import UploadFile

from app.services import storage_service


def _upload(data: bytes, filename: str) -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


@pytest.fixture(autouse=True)
def _isolated_upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_service, "UPLOAD_DIR", str(tmp_path))
    # Don't let the opportunistic sweep touch anything mid-test.
    monkeypatch.setattr(storage_service, "_maybe_sweep_stale_uploads", _noop)


async def _noop():
    return None


# ── Accept: valid UTF-8 .txt ─────────────────────────────────────────────────

async def test_accepts_plain_utf8_text():
    doc = await storage_service.save_upload(_upload("hello world — café".encode("utf-8"), "notes.txt"))
    assert doc.filename == "notes.txt"


async def test_accepts_utf8_bom_text():
    doc = await storage_service.save_upload(_upload(b"\xef\xbb\xbfhello", "bom.txt"))
    assert doc.filename == "bom.txt"


# ── Reject: unsupported types, even with genuine content ────────────────────

async def test_rejects_real_pdf_no_extractor_exists():
    # A REAL PDF (genuine %PDF- header) must be rejected at the extension gate:
    # there is no server-side extractor, so accepting it would just defer the
    # failure to scan creation as a masked 500.
    with pytest.raises(ValueError, match="not allowed"):
        await storage_service.save_upload(_upload(b"%PDF-1.4\n...rest of pdf...", "doc.pdf"))


async def test_rejects_real_docx_no_extractor_exists():
    # Same for a genuine DOCX (ZIP local-file-header signature).
    with pytest.raises(ValueError, match="not allowed"):
        await storage_service.save_upload(_upload(b"PK\x03\x04" + b"\x00" * 20, "doc.docx"))


async def test_rejects_text_renamed_to_pdf():
    with pytest.raises(ValueError, match="not allowed"):
        await storage_service.save_upload(_upload(b"just plain text, not a pdf", "fake.pdf"))


async def test_rejects_extension_not_allowed():
    with pytest.raises(ValueError, match="not allowed"):
        await storage_service.save_upload(_upload(b"anything", "script.exe"))


# ── Reject: bad content behind a .txt name ──────────────────────────────────

async def test_rejects_binary_renamed_to_txt():
    with pytest.raises(ValueError, match="does not match a text file"):
        await storage_service.save_upload(_upload(b"%PDF-1.4\xff\xfe\x00binary junk", "fake.txt"))


async def test_rejects_nul_bytes_in_txt():
    with pytest.raises(ValueError, match="does not match a text file"):
        await storage_service.save_upload(_upload(b"hello\x00world", "fake.txt"))


async def test_rejects_non_utf8_txt():
    with pytest.raises(ValueError, match="does not match a text file"):
        # Latin-1-only byte (0xe9 = 'é' in latin-1) is not valid standalone UTF-8.
        await storage_service.save_upload(_upload(b"caf\xe9", "latin1.txt"))
