import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse
from app.routes import documents, scans, reports, rewrites, auth, payments, translate, feedback, keys, ext
from app.models.db import init_db, async_session
from app.config import COOKIE_SECURE, SECRET_KEY, FRONTEND_URL
from sqlalchemy import text as sa_text

logger = logging.getLogger("draftproof")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="DraftProof API", version="1.0.0", lifespan=lifespan)

# ── Global exception handlers (never leak raw errors to users) ──

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    # Only redirect OAuth callback routes (google/callback, microsoft/callback) — NOT /me or /logout
    auth_api_paths = ("/api/auth/google/", "/api/auth/microsoft/")
    if request.url.path.startswith(auth_api_paths):
        return _auth_error_redirect(exc.status_code, exc.detail or "Authentication failed")
    # Pass through domain-specific error messages (e.g. "Insufficient tokens")
    # instead of replacing them with generic text
    detail = exc.detail if exc.detail and len(exc.detail) < 200 else _friendly_message(exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    logger.warning("Validation error on %s: %s", request.url.path, errors)
    # Return actual validation details so frontend can show useful info
    return JSONResponse(status_code=400, content={
        "detail": "Invalid request.",
        "errors": [{"loc": e.get("loc", []), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in errors],
    })


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    if request.url.path.startswith("/api/auth/"):
        return _auth_error_redirect(500, "Something went wrong during sign-in")
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again."})


def _auth_error_redirect(status_code: int, detail: str) -> JSONResponse:
    """Redirect OAuth errors to the frontend sign-in page with a friendly message."""
    from starlette.responses import RedirectResponse
    msg = _friendly_message(status_code)
    return RedirectResponse(url=f"{FRONTEND_URL}/signin?error={msg}", status_code=302)


def _friendly_message(status_code: int) -> str:
    return {
        400: "Invalid request. Please try again.",
        401: "Session expired. Please sign in again.",
        403: "Access denied. Your email domain is not supported.",
        404: "The requested resource was not found.",
        429: "Too many requests. Please wait a moment.",
        500: "Something went wrong. Please try again.",
    }.get(status_code, "An unexpected error occurred.")


frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
# Support multiple origins (e.g. draftproof.app + www.draftproof.app)
allowed_origins = [o.strip() for o in frontend_url.split(",") if o.strip()]

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="session",
    same_site="lax",
    https_only=COOKIE_SECURE,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    # Authorization / X-API-Key let the MS Word task pane (a browser fetch) send
    # its API key cross-origin; the web SPA still uses cookies.
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(scans.router, prefix="/api/scans", tags=["scans"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(rewrites.router, prefix="/api/rewrites", tags=["rewrites"])
app.include_router(payments.router, prefix="/api/payments", tags=["payments"])
app.include_router(translate.router, prefix="/api/translate", tags=["translate"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["feedback"])
app.include_router(keys.router, prefix="/api/keys", tags=["api-keys"])
app.include_router(ext.router, prefix="/api/ext", tags=["extension"])


@app.get("/api/health")
async def health():
    db_ok = False
    try:
        async with async_session() as session:
            await session.execute(sa_text("SELECT 1"))
        db_ok = True
    except Exception:
        pass
    if not db_ok:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "db": "down"})
    return {"status": "ok", "db": "up"}


# Serve frontend static files (production only)
static_path = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(static_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(static_path, "assets")), name="assets")

    @app.get("/report/{report_id}/rewrite")
    async def legacy_rewrite_route(report_id: str, request: Request):
        rewrite_id = request.query_params.get("rid")
        if rewrite_id:
            return RedirectResponse(url=f"/rewrite/{rewrite_id}", status_code=302)
        return RedirectResponse(url=f"/report/{report_id}", status_code=302)

    @app.api_route("/{path:path}", methods=["GET", "HEAD"])
    async def serve_spa(path: str):
        redirect_target = legacy_frontend_redirect_target(path)
        if redirect_target:
            return RedirectResponse(url=redirect_target, status_code=301)

        file_path, cache_control, status_code, extra_headers = resolve_frontend_file(static_path, path)
        headers = dict(extra_headers)
        if cache_control:
            headers["Cache-Control"] = cache_control
        return FileResponse(file_path, status_code=status_code, headers=headers or None)


PRIVATE_FRONTEND_PREFIXES = (
    "dashboard",
    "scan",
    "reports",
    "report/",
    "rewrite/",
    "buy",
    "history",
    "api-keys",
    "auth/callback",
)


LEGACY_FRONTEND_REDIRECTS = {
    "essay-checker": "/content-checker",
    "zh/essay-checker": "/zh/content-checker",
}


def legacy_frontend_redirect_target(path: str) -> str | None:
    normalized = os.path.normpath(path.strip("/"))
    if normalized.startswith("..") or os.path.isabs(normalized):
        return None
    return LEGACY_FRONTEND_REDIRECTS.get(normalized)


def resolve_frontend_file(root_path: str, path: str) -> tuple[str, str | None, int, dict[str, str]]:
    normalized = os.path.normpath(path.strip("/"))
    if normalized in {"", "."}:
        return os.path.join(root_path, "index.html"), "no-cache", 200, {}
    elif normalized.startswith("..") or os.path.isabs(normalized):
        return frontend_not_found_file(root_path), "no-cache", 404, {"X-Robots-Tag": "noindex, nofollow"}

    file_path = os.path.join(root_path, normalized)
    if normalized and os.path.isfile(file_path):
        # Add-in files (word-addin/*) have FIXED names (no content hash), so without an
        # explicit directive Office/Word heuristically caches stale copies indefinitely
        # (the "I refreshed, nothing changed" bug). Force revalidation so updates show on
        # reload; the ETag/Last-Modified make this a cheap 304 when unchanged. Hashed
        # frontend assets are mounted separately at /assets and keep long caching.
        cache_control = "no-cache" if normalized.startswith("word-addin/") else None
        return file_path, cache_control, 200, {}

    route_index_path = os.path.join(file_path, "index.html")
    if normalized and os.path.isfile(route_index_path):
        return route_index_path, "no-cache", 200, {}

    if is_private_frontend_path(normalized):
        return os.path.join(root_path, "index.html"), "no-cache", 200, {"X-Robots-Tag": "noindex, nofollow"}

    return frontend_not_found_file(root_path), "no-cache", 404, {"X-Robots-Tag": "noindex, nofollow"}


def is_private_frontend_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in PRIVATE_FRONTEND_PREFIXES)


def frontend_not_found_file(root_path: str) -> str:
    not_found_path = os.path.join(root_path, "404", "index.html")
    if os.path.isfile(not_found_path):
        return not_found_path
    return os.path.join(root_path, "index.html")
