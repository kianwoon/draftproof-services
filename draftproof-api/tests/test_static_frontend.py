from pathlib import Path

from app.main import legacy_frontend_redirect_target, resolve_frontend_file


def test_resolve_frontend_file_serves_home_for_root_path(tmp_path):
    root = tmp_path
    (root / "index.html").write_text("home", encoding="utf-8")

    file_path, cache_control, status_code, headers = resolve_frontend_file(str(root), "")

    assert Path(file_path) == root / "index.html"
    assert cache_control == "no-cache"
    assert status_code == 200
    assert headers == {}


def test_resolve_frontend_file_uses_prerendered_route_index(tmp_path):
    root = tmp_path
    (root / "index.html").write_text("home", encoding="utf-8")
    pricing_dir = root / "pricing"
    pricing_dir.mkdir()
    (pricing_dir / "index.html").write_text("pricing", encoding="utf-8")

    file_path, cache_control, status_code, headers = resolve_frontend_file(str(root), "pricing")

    assert Path(file_path) == pricing_dir / "index.html"
    assert cache_control == "no-cache"
    assert status_code == 200
    assert headers == {}


def test_resolve_frontend_file_keeps_exact_static_file(tmp_path):
    root = tmp_path
    (root / "index.html").write_text("home", encoding="utf-8")
    (root / "sitemap.xml").write_text("<urlset />", encoding="utf-8")

    file_path, cache_control, status_code, headers = resolve_frontend_file(str(root), "sitemap.xml")

    assert Path(file_path) == root / "sitemap.xml"
    assert cache_control is None
    assert status_code == 200
    assert headers == {}


def test_legacy_content_checker_routes_redirect_to_canonical_paths():
    assert legacy_frontend_redirect_target("essay-checker") == "/content-checker"
    assert legacy_frontend_redirect_target("zh/essay-checker") == "/zh/content-checker"


def test_legacy_frontend_redirect_rejects_parent_directory_paths():
    assert legacy_frontend_redirect_target("../essay-checker") is None


def test_resolve_frontend_file_serves_private_route_with_noindex(tmp_path):
    root = tmp_path
    (root / "index.html").write_text("home", encoding="utf-8")

    file_path, cache_control, status_code, headers = resolve_frontend_file(str(root), "dashboard")

    assert Path(file_path) == root / "index.html"
    assert cache_control == "no-cache"
    assert status_code == 200
    assert headers == {"X-Robots-Tag": "noindex, nofollow"}


def test_api_keys_route_is_served_as_private_spa_route(tmp_path):
    """/api-keys is a private (ProtectedRoute) SPA page, not prerendered — a direct
    visit/refresh must serve index.html so React Router handles it, not 404."""
    root = tmp_path
    (root / "index.html").write_text("home", encoding="utf-8")

    file_path, cache_control, status_code, headers = resolve_frontend_file(str(root), "api-keys")

    assert Path(file_path) == root / "index.html"
    assert status_code == 200
    assert headers == {"X-Robots-Tag": "noindex, nofollow"}


def test_resolve_frontend_file_returns_404_for_unknown_route(tmp_path):
    root = tmp_path
    (root / "index.html").write_text("home", encoding="utf-8")
    not_found_dir = root / "404"
    not_found_dir.mkdir()
    (not_found_dir / "index.html").write_text("not found", encoding="utf-8")

    file_path, cache_control, status_code, headers = resolve_frontend_file(str(root), "unknown")

    assert Path(file_path) == not_found_dir / "index.html"
    assert cache_control == "no-cache"
    assert status_code == 404
    assert headers == {"X-Robots-Tag": "noindex, nofollow"}


def test_resolve_frontend_file_rejects_parent_directory_paths(tmp_path):
    root = tmp_path / "static"
    root.mkdir()
    (root / "index.html").write_text("home", encoding="utf-8")
    not_found_dir = root / "404"
    not_found_dir.mkdir()
    (not_found_dir / "index.html").write_text("not found", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")

    file_path, cache_control, status_code, headers = resolve_frontend_file(str(root), "../secret.txt")

    assert Path(file_path) == not_found_dir / "index.html"
    assert cache_control == "no-cache"
    assert status_code == 404
    assert headers == {"X-Robots-Tag": "noindex, nofollow"}
