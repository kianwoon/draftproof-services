"""Security regression tests for _read_document_text_sync path traversal (finding H2)
and upload-consumption lifecycle (delete only after the scan job commit)."""

import uuid

from app.services import scan_service


def test_read_document_rejects_path_traversal(tmp_path, monkeypatch):
    # Plant a "secret" file OUTSIDE the upload dir, then try to read it via a
    # traversal document_id. The fix must refuse (return no content) rather than leak it.
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    upload = tmp_path / "uploads"
    upload.mkdir()
    monkeypatch.setattr(scan_service, "UPLOAD_DIR", str(upload))

    # os.path.join(upload, "../secret" + ".txt") resolves to the planted secret.
    assert scan_service._read_document_text_sync("../secret") == ("", None)


def test_read_document_rejects_non_uuid(tmp_path, monkeypatch):
    upload = tmp_path / "uploads"
    upload.mkdir()
    monkeypatch.setattr(scan_service, "UPLOAD_DIR", str(upload))

    assert scan_service._read_document_text_sync("../../../../etc/passwd") == ("", None)
    assert scan_service._read_document_text_sync("not-a-uuid") == ("", None)
    assert scan_service._read_document_text_sync("") == ("", None)


def test_read_document_reads_valid_uuid(tmp_path, monkeypatch):
    upload = tmp_path / "uploads"
    upload.mkdir()
    monkeypatch.setattr(scan_service, "UPLOAD_DIR", str(upload))

    doc_id = str(uuid.uuid4())
    path = upload / f"{doc_id}.txt"
    path.write_text("hello world", encoding="utf-8")
    assert scan_service._read_document_text_sync(doc_id) == ("hello world", str(path))


def test_read_document_does_not_delete_on_extraction(tmp_path, monkeypatch):
    """Extraction must NOT delete the upload: create_scan can still fail AFTER
    extraction (empty text, no credit account, insufficient tokens), and a user who
    tops up and retries the same document_id must find their document intact.
    Deletion happens in create_scan only after the job commit, via
    _delete_upload_best_effort on the returned path."""
    upload = tmp_path / "uploads"
    upload.mkdir()
    monkeypatch.setattr(scan_service, "UPLOAD_DIR", str(upload))

    doc_id = str(uuid.uuid4())
    path = upload / f"{doc_id}.txt"
    path.write_text("read me twice", encoding="utf-8")

    assert scan_service._read_document_text_sync(doc_id) == ("read me twice", str(path))
    assert path.exists()
    # A retried call (e.g. after topping up credits) still finds the document.
    assert scan_service._read_document_text_sync(doc_id) == ("read me twice", str(path))


def test_legacy_binary_upload_is_unreadable_not_a_500(tmp_path, monkeypatch):
    """A REAL binary PDF already on disk (uploaded before ALLOWED_EXTENSIONS was
    narrowed to .txt) must resolve to no-content — which create_scan turns into the
    clean 400 'Document text not found or empty' — instead of an uncaught
    UnicodeDecodeError that used to surface as a masked 500."""
    upload = tmp_path / "uploads"
    upload.mkdir()
    monkeypatch.setattr(scan_service, "UPLOAD_DIR", str(upload))

    doc_id = str(uuid.uuid4())
    # Genuine binary PDF-ish bytes: valid %PDF- header followed by non-UTF-8 stream bytes.
    (upload / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4\n\xff\xfe\x00\x81binary stream \xe9\x00")

    assert scan_service._read_document_text_sync(doc_id) == ("", None)


def test_delete_upload_best_effort_removes_file(tmp_path):
    path = tmp_path / f"{uuid.uuid4()}.txt"
    path.write_text("consume me once", encoding="utf-8")

    scan_service._delete_upload_best_effort(str(path))
    assert not path.exists()


def test_delete_upload_best_effort_swallows_failure(tmp_path, monkeypatch):
    """A delete failure (e.g. permissions) must not raise -- deletion is hygiene,
    not correctness; only a warning is logged."""
    path = tmp_path / f"{uuid.uuid4()}.txt"
    path.write_text("still here", encoding="utf-8")

    monkeypatch.setattr(
        scan_service.os, "remove", lambda p: (_ for _ in ()).throw(OSError("locked"))
    )
    scan_service._delete_upload_best_effort(str(path))  # must not raise
