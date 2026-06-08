"""In-app feedback → GitHub Issues.

Visitors on draftproof.app can report a bug or suggest a feature from a floating
widget. Because the GitHub repo is *private*, users cannot file issues directly,
so this endpoint files them on their behalf with a fine-grained PAT scoped to
Issues:write only. A Cloudflare Turnstile token is verified server-side first so
the public endpoint can't be turned into a GitHub-issue spam cannon.

Auth is optional: anyone may submit (gated by Turnstile, not login), but if the
caller happens to be signed in we attach their email to the issue for follow-up.
"""

import logging
from typing import Literal, Optional

import httpx
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from jose import jwt

from app.config import (
    FEEDBACK_GITHUB_TOKEN,
    FEEDBACK_GITHUB_REPO,
    TURNSTILE_SECRET_KEY,
    TURNSTILE_SITE_KEY,
    SECRET_KEY,
    JWT_ALGORITHM,
)

router = APIRouter()
log = logging.getLogger("feedback")

TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
GITHUB_API = "https://api.github.com"

# Map the user-facing kind to GitHub labels. "user-feedback" tags every issue so
# you can filter inbound reports from your own backlog at a glance.
_LABELS = {
    "bug": ["bug", "user-feedback"],
    "feature": ["enhancement", "user-feedback"],
}


class FeedbackIn(BaseModel):
    type: Literal["bug", "feature"]
    title: str = Field(..., min_length=3, max_length=120)
    body: str = Field(..., min_length=10, max_length=5000)
    email: Optional[str] = Field(None, max_length=254)
    page_url: Optional[str] = Field(None, max_length=500)
    turnstile_token: str = Field(..., min_length=1, max_length=2048)


class FeedbackOut(BaseModel):
    ok: bool
    url: Optional[str] = None


class FeedbackConfigOut(BaseModel):
    # Public values only — safe to expose to the browser.
    turnstile_site_key: str
    enabled: bool


@router.get("/config", response_model=FeedbackConfigOut)
async def feedback_config():
    """Runtime public config for the feedback widget. Lets the SITE key live as a
    plain Koyeb env var (read here) instead of a Vite build-time inline."""
    enabled = bool(FEEDBACK_GITHUB_TOKEN and TURNSTILE_SECRET_KEY and TURNSTILE_SITE_KEY)
    return FeedbackConfigOut(turnstile_site_key=TURNSTILE_SITE_KEY, enabled=enabled)


def _optional_user(request: Request) -> Optional[dict]:
    """Best-effort: return the signed-in user, or None. Never raises.

    Mirrors ``get_current_user`` in routes/auth.py but degrades to anonymous
    instead of 401 — feedback is open to everyone, login just enriches it.
    """
    token = request.cookies.get("token")
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return {"id": payload.get("sub"), "email": payload.get("email")}
    except jwt.JWTError:
        return None


def _client_ip(request: Request) -> Optional[str]:
    """Resolve the real visitor IP behind Cloudflare/Koyeb proxies for Turnstile."""
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


async def _verify_turnstile(token: str, remoteip: Optional[str]) -> bool:
    data = {"secret": TURNSTILE_SECRET_KEY, "response": token}
    if remoteip:
        data["remoteip"] = remoteip
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(TURNSTILE_VERIFY_URL, data=data)
            resp.raise_for_status()
            result = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Turnstile verify failed to reach Cloudflare: %s", exc)
        raise HTTPException(status_code=502, detail="Verification service unavailable") from exc
    return bool(result.get("success"))


# ── LEARNING CONTRIBUTION POINT ──────────────────────────────────────────────
# This is the one piece with a real design tradeoff worth your input: what
# metadata goes into the issue body. More context = easier triage/repro, but
# also more noise and potential PII landing in your repo. A working default is
# implemented below — tune it to your preference (see the note after this file).
def _build_issue_body(payload: FeedbackIn, submitter: Optional[dict]) -> str:
    """Format the GitHub issue body from the submission + optional submitter.

    Tradeoff to consider:
      - Include submitter email so you can follow up? (PII in a private repo)
      - Include page_url for repro context? (usually safe + very useful)
      - How much to trust/escape user text? (it lands as Markdown verbatim)
    """
    contact = (submitter or {}).get("email") or payload.email or "anonymous"
    lines = [
        payload.body.strip(),
        "",
        "---",
        f"- **Type:** {payload.type}",
        f"- **Submitted via:** draftproof.app feedback widget",
        f"- **Contact:** {contact}",
    ]
    if payload.page_url:
        lines.append(f"- **Page:** {payload.page_url}")
    return "\n".join(lines)
# ─────────────────────────────────────────────────────────────────────────────


@router.post("", response_model=FeedbackOut)
async def submit_feedback(body: FeedbackIn, request: Request):
    if not (FEEDBACK_GITHUB_TOKEN and FEEDBACK_GITHUB_REPO and TURNSTILE_SECRET_KEY):
        log.error("Feedback endpoint hit but env vars are not configured")
        raise HTTPException(status_code=503, detail="Feedback is not configured")

    if not await _verify_turnstile(body.turnstile_token, _client_ip(request)):
        raise HTTPException(status_code=400, detail="Anti-bot verification failed. Please retry.")

    submitter = _optional_user(request)
    prefix = "Bug" if body.type == "bug" else "Feature"
    issue = {
        "title": f"[{prefix}] {body.title.strip()}",
        "body": _build_issue_body(body, submitter),
        "labels": _LABELS[body.type],
    }
    headers = {
        "Authorization": f"Bearer {FEEDBACK_GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "draftproof-feedback",
    }
    url = f"{GITHUB_API}/repos/{FEEDBACK_GITHUB_REPO}/issues"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=issue, headers=headers)
            resp.raise_for_status()
            created = resp.json()
    except httpx.HTTPStatusError as exc:
        log.error("GitHub issue create failed: %s %s", exc.response.status_code, exc.response.text[:300])
        raise HTTPException(status_code=502, detail="Could not file your feedback. Please try again later.") from exc
    except (httpx.HTTPError, ValueError) as exc:
        log.error("GitHub issue create transport error: %s", exc)
        raise HTTPException(status_code=502, detail="Could not file your feedback. Please try again later.") from exc

    return FeedbackOut(ok=True, url=created.get("html_url"))
