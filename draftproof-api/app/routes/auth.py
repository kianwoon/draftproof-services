from datetime import datetime, timedelta
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from authlib.integrations.starlette_client import OAuth
from jose import jwt

from app.config import (
    SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRATION_HOURS,
    ALLOWED_EMAIL_DOMAINS, FRONTEND_URL,
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
    MICROSOFT_CLIENT_ID, MICROSOFT_CLIENT_SECRET, MICROSOFT_TENANT,
)
from app.models.db import get_db, User

router = APIRouter()

oauth = OAuth()

# Google OAuth
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# Microsoft OAuth
oauth.register(
    name="microsoft",
    client_id=MICROSOFT_CLIENT_ID,
    client_secret=MICROSOFT_CLIENT_SECRET,
    server_metadata_url=f"https://login.microsoftonline.com/{MICROSOFT_TENANT}/v2.0/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile User.Read"},
)


def _build_callback_url(request: Request, callback_name: str) -> str:
    url = str(request.url_for(callback_name))
    # Koyeb terminates SSL at LB — force https for OAuth redirect URIs
    if url.startswith("http://"):
        url = "https://" + url[7:]
    return url


def _validate_email_domain(email: str) -> str:
    domain = email.split("@")[-1].lower()
    if domain not in ALLOWED_EMAIL_DOMAINS:
        raise HTTPException(
            status_code=403,
            detail=f"Email domain '{domain}' is not allowed. Please use Gmail or Hotmail.",
        )
    return email


def _create_jwt(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def _upsert_user(db: Session, provider: str, user_info: dict) -> User:
    email = user_info["email"].lower()
    provider_id = user_info.get("sub", user_info.get("id", ""))

    user = db.query(User).filter(User.email == email).first()
    if user:
        user.last_login = datetime.utcnow()
        user.display_name = user_info.get("name", user.display_name)
        user.avatar_url = user_info.get("picture", user.avatar_url)
        user.provider = provider
        user.provider_id = provider_id
    else:
        user = User(
            email=email,
            display_name=user_info.get("name", ""),
            provider=provider,
            provider_id=provider_id,
            avatar_url=user_info.get("picture"),
        )
        db.add(user)

    db.commit()
    db.refresh(user)
    return user


@router.get("/google")
async def auth_google(request: Request):
    redirect_uri = _build_callback_url(request, "auth_google_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback")
async def auth_google_callback(request: Request, db: Session = Depends(get_db)):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo") or await oauth.google.userinfo(token=token)

    _validate_email_domain(user_info["email"])
    user = _upsert_user(db, "google", user_info)
    jwt_token = _create_jwt(user.id, user.email)

    response = RedirectResponse(url=f"{FRONTEND_URL}/auth/callback")
    response.set_cookie(
        "token", jwt_token,
        httponly=True, secure=True, samesite="lax",
        max_age=JWT_EXPIRATION_HOURS * 3600,
    )
    return response


@router.get("/microsoft")
async def auth_microsoft(request: Request):
    redirect_uri = _build_callback_url(request, "auth_microsoft_callback")
    return await oauth.microsoft.authorize_redirect(request, redirect_uri)


@router.get("/microsoft/callback")
async def auth_microsoft_callback(request: Request, db: Session = Depends(get_db)):
    token = await oauth.microsoft.authorize_access_token(request)
    user_info = token.get("userinfo") or await oauth.microsoft.userinfo(token=token)

    _validate_email_domain(user_info["email"])
    user = _upsert_user(db, "microsoft", user_info)
    jwt_token = _create_jwt(user.id, user.email)

    response = RedirectResponse(url=f"{FRONTEND_URL}/auth/callback")
    response.set_cookie(
        "token", jwt_token,
        httponly=True, secure=True, samesite="lax",
        max_age=JWT_EXPIRATION_HOURS * 3600,
    )
    return response


@router.get("/me")
async def get_me(request: Request):
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return {"id": payload["sub"], "email": payload["email"]}
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/logout")
async def logout():
    response = RedirectResponse(url=FRONTEND_URL)
    response.delete_cookie("token")
    return response
