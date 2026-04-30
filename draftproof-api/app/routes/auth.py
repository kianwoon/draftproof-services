from datetime import datetime, timedelta
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from authlib.integrations.starlette_client import OAuth
from jose import jwt

from app.config import (
    SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRATION_HOURS,
    ALLOWED_EMAIL_DOMAINS, FRONTEND_URL,
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
    MICROSOFT_CLIENT_ID, MICROSOFT_CLIENT_SECRET, MICROSOFT_TENANT,
)
from app.models.db import get_db, User, UserIdentity, CreditAccount

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
        "sub": str(user_id),
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


async def _upsert_user(db: AsyncSession, provider: str, user_info: dict) -> User:
    email = user_info["email"].lower().strip()
    email_normalized = email
    provider_user_id = str(user_info.get("sub", user_info.get("id", "")))
    provider_email_verified = user_info.get("email_verified", True)

    # Find existing identity by provider + provider_user_id
    result = await db.execute(
        select(UserIdentity).where(
            UserIdentity.provider == provider,
            UserIdentity.provider_user_id == provider_user_id,
        )
    )
    identity = result.scalar_one_or_none()

    if identity:
        # Existing user — update
        identity.last_login_at = datetime.utcnow()
        identity.provider_email = email
        identity.provider_email_verified = provider_email_verified

        result = await db.execute(select(User).where(User.id == identity.user_id))
        user = result.scalar_one()
        user.display_name = user_info.get("name", user.display_name)
        user.avatar_url = user_info.get("picture", user.avatar_url)
        user.updated_at = datetime.utcnow()
    else:
        # Check if user exists by email
        result = await db.execute(
            select(User).where(User.email_normalized == email_normalized)
        )
        user = result.scalar_one_or_none()

        if not user:
            # Create new user
            user = User(
                email=email,
                email_normalized=email_normalized,
                display_name=user_info.get("name", ""),
                avatar_url=user_info.get("picture"),
                status="active",
            )
            db.add(user)
            await db.flush()

            # Create credit account
            account = CreditAccount(user_id=user.id)
            db.add(account)

        # Create identity
        identity = UserIdentity(
            user_id=user.id,
            provider=provider,
            provider_user_id=provider_user_id,
            provider_email=email,
            provider_email_verified=provider_email_verified,
            last_login_at=datetime.utcnow(),
        )
        db.add(identity)

    await db.commit()
    await db.refresh(user)
    return user


@router.get("/google")
async def auth_google(request: Request):
    redirect_uri = _build_callback_url(request, "auth_google_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback")
async def auth_google_callback(request: Request, db: AsyncSession = Depends(get_db)):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo")
    if not user_info:
        user_info = await oauth.google.userinfo(token=token)

    _validate_email_domain(user_info["email"])
    user = await _upsert_user(db, "google", user_info)
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
async def auth_microsoft_callback(request: Request, db: AsyncSession = Depends(get_db)):
    import logging
    logger = logging.getLogger("auth.microsoft")
    try:
        token = await oauth.microsoft.authorize_access_token(
            request, claims_options={"iss": {"essential": False}}
        )
        logger.info("Microsoft token keys: %s", list(token.keys()))
        user_info = token.get("userinfo")
        if not user_info:
            # Microsoft returns user info in the ID token claims, not as a top-level key
            id_token = token.get("id_token")
            if id_token and isinstance(id_token, dict):
                user_info = id_token
            else:
                user_info = await oauth.microsoft.userinfo(token=token)
            logger.info("Userinfo keys: %s", list(user_info.keys()) if user_info else "None")

        email = user_info.get("email") or user_info.get("preferred_username") or user_info.get("upn")
        if not email:
            raise HTTPException(status_code=400, detail="Could not retrieve email from Microsoft account")

        user_info["email"] = email
        _validate_email_domain(email)
        user = await _upsert_user(db, "microsoft", user_info)
        jwt_token = _create_jwt(user.id, user.email)

        response = RedirectResponse(url=f"{FRONTEND_URL}/auth/callback")
        response.set_cookie(
            "token", jwt_token,
            httponly=True, secure=True, samesite="lax",
            max_age=JWT_EXPIRATION_HOURS * 3600,
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.getLogger("auth.microsoft").error("Microsoft OAuth callback failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Microsoft sign-in failed: {str(e)}")


async def get_current_user(request: Request) -> dict:
    """Dependency: extracts user_id from JWT cookie."""
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return {"id": payload["sub"], "email": payload["email"]}
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


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
    from starlette.responses import JSONResponse
    response = JSONResponse({"ok": True})
    response.delete_cookie("token", httponly=True, secure=True, samesite="lax")
    return response
