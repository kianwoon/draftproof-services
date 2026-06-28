"""Security regression tests for _read_document_text_sync path traversal (finding H2)."""

import uuid

from app.services import scan_service


def test_read_document_rejects_path_traversal(tmp_path, monkeypatch):
    # Plant a "secret" file OUTSIDE the upload dir, then try to read it via a
    # traversal document_id. The fix must refuse (return "") rather than leak it.
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    upload = tmp_path / "uploads"
    upload.mkdir()
    monkeypatch.setattr(scan_service, "UPLOAD_DIR", str(upload))

    # os.path.join(upload, "../secret" + ".txt") resolves to the planted secret.
    assert scan_service._read_document_text_sync("../secret") == ""


def test_read_document_rejects_non_uuid(tmp_path, monkeypatch):
    upload = tmp_path / "uploads"
    upload.mkdir()
    monkeypatch.setattr(scan_service, "UPLOAD_DIR", str(upload))

    assert scan_service._read_document_text_sync("../../../../etc/passwd") == ""
    assert scan_service._read_document_text_sync("not-a-uuid") == ""
    assert scan_service._read_document_text_sync("") == ""


def test_read_document_reads_valid_uuid(tmp_path, monkeypatch):
    upload = tmp_path / "uploads"
    upload.mkdir()
    monkeypatch.setattr(scan_service, "UPLOAD_DIR", str(upload))

    doc_id = str(uuid.uuid4())
    (upload / f"{doc_id}.txt").write_text("hello world", encoding="utf-8")
    assert scan_service._read_document_text_sync(doc_id) == "hello world"
