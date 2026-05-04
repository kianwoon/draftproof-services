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
from app.routes import documents, scans, reports, rewrites, auth, payments
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
    # All other routes: return JSON
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": _friendly_message(exc.status_code)},
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
    allow_headers=["Content-Type"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(scans.router, prefix="/api/scans", tags=["scans"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(rewrites.router, prefix="/api/rewrites", tags=["rewrites"])
app.include_router(payments.router, prefix="/api/payments", tags=["payments"])


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

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        file_path = os.path.join(static_path, path)
        if path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(
            os.path.join(static_path, "index.html"),
            headers={"Cache-Control": "no-cache"},
        )
