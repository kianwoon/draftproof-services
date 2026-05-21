from pathlib import Path

from app.main import resolve_frontend_file


def test_resolve_frontend_file_uses_prerendered_route_index(tmp_path):
    root = tmp_path
    (root / "index.html").write_text("home", encoding="utf-8")
    pricing_dir = root / "pricing"
    pricing_dir.mkdir()
    (pricing_dir / "index.html").write_text("pricing", encoding="utf-8")

    file_path, cache_control = resolve_frontend_file(str(root), "pricing")

    assert Path(file_path) == pricing_dir / "index.html"
    assert cache_control == "no-cache"


def test_resolve_frontend_file_keeps_exact_static_file(tmp_path):
    root = tmp_path
    (root / "index.html").write_text("home", encoding="utf-8")
    (root / "sitemap.xml").write_text("<urlset />", encoding="utf-8")

    file_path, cache_control = resolve_frontend_file(str(root), "sitemap.xml")

    assert Path(file_path) == root / "sitemap.xml"
    assert cache_control is None


def test_resolve_frontend_file_falls_back_to_spa_index(tmp_path):
    root = tmp_path
    (root / "index.html").write_text("home", encoding="utf-8")

    file_path, cache_control = resolve_frontend_file(str(root), "unknown")

    assert Path(file_path) == root / "index.html"
    assert cache_control == "no-cache"


def test_resolve_frontend_file_rejects_parent_directory_paths(tmp_path):
    root = tmp_path / "static"
    root.mkdir()
    (root / "index.html").write_text("home", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")

    file_path, cache_control = resolve_frontend_file(str(root), "../secret.txt")

    assert Path(file_path) == root / "index.html"
    assert cache_control == "no-cache"
