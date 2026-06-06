"""Lightweight text translation proxy.

The frontend submitted-content editor lets ESL (e.g. Chinese) students draft a
passage in their first language, highlight it, and translate the selection to
English as a *starting draft they then edit*. We proxy Google's public
``translate_a/single`` endpoint server-side because that endpoint sends no CORS
headers, so a direct browser call would be blocked. Keeping it on the API also
lets us swap providers later without touching the UI.
"""

import httpx
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.routes.auth import get_current_user

router = APIRouter()

GOOGLE_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"


class TranslateIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    source: str = Field("auto", max_length=10)
    target: str = Field("en", max_length=10)


class TranslateOut(BaseModel):
    text: str
    detected_source: str | None = None


@router.post("", response_model=TranslateOut)
async def translate_text(body: TranslateIn, user: dict = Depends(get_current_user)):
    params = {
        "client": "gtx",
        "sl": body.source or "auto",
        "tl": body.target or "en",
        "dt": "t",
        "q": body.text,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(GOOGLE_TRANSLATE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Translation service unavailable") from exc

    # Response shape: [ [ [translated, original, ...], ... ], ..., detected_lang, ... ]
    segments = data[0] if isinstance(data, list) and data and isinstance(data[0], list) else []
    translated = "".join(seg[0] for seg in segments if seg and seg[0])
    detected = data[2] if isinstance(data, list) and len(data) > 2 and isinstance(data[2], str) else None

    if not translated.strip():
        raise HTTPException(status_code=502, detail="Translation returned no text")

    return TranslateOut(text=translated.strip(), detected_source=detected)
