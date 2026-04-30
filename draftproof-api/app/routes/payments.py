import stripe
from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import (
    STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET,
    TOKEN_PRICE_SGD, TOKEN_PACKS, FRONTEND_URL, SECRET_KEY, JWT_ALGORITHM,
)
from app.models.db import get_db, User, CreditAccount, CreditLedger, Payment
from jose import jwt, JWTError

router = APIRouter()
stripe.api_key = STRIPE_SECRET_KEY


def _get_user_id(request: Request) -> str:
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.get("/packs")
async def get_packs():
    return [
        {
            "id": key,
            "name": pack["name"],
            "tokens": pack["tokens"],
            "price_sgd": round(pack["tokens"] * TOKEN_PRICE_SGD, 2),
        }
        for key, pack in TOKEN_PACKS.items()
    ]


@router.post("/checkout")
async def create_checkout(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = _get_user_id(request)
    body = await request.json()
    pack_id = body.get("pack_id")

    if pack_id not in TOKEN_PACKS:
        raise HTTPException(status_code=400, detail="Invalid pack")

    pack = TOKEN_PACKS[pack_id]
    price_sgd = round(pack["tokens"] * TOKEN_PRICE_SGD, 2)
    price_cents = int(price_sgd * 100)

    session = stripe.checkout.Session.create(
        mode="payment",
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "sgd",
                "product_data": {"name": f"DraftProof — {pack['name']} ({pack['tokens']} tokens)"},
                "unit_amount": price_cents,
            },
            "quantity": 1,
        }],
        metadata={"user_id": user_id, "pack_id": pack_id, "tokens": str(pack["tokens"])},
        success_url=f"{FRONTEND_URL}/buy?success=1",
        cancel_url=f"{FRONTEND_URL}/buy?canceled=1",
    )

    return {"url": session.url}


@router.get("/balance")
async def get_balance(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = _get_user_id(request)
    result = await db.execute(
        select(CreditAccount).where(CreditAccount.user_id == user_id)
    )
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(status_code=404, detail="No credit account")
    return {"balance": account.balance_tokens, "reserved": account.reserved_tokens}


@router.post("/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event["type"] == "checkout.session.completed":
        session_data = event["data"]["object"]
        user_id = session_data["metadata"]["user_id"]
        tokens = int(session_data["metadata"]["tokens"])
        stripe_session_id = session_data["id"]
        amount_cents = session_data["amount_total"]

        # Check idempotency — skip if already processed
        result = await db.execute(
            select(Payment).where(Payment.provider_payment_id == stripe_session_id)
        )
        if result.scalar_one_or_none():
            return {"ok": True, "skipped": True}

        # Credit tokens
        result = await db.execute(
            select(CreditAccount).where(CreditAccount.user_id == user_id)
        )
        account = result.scalar_one_or_none()
        if not account:
            raise HTTPException(status_code=404, detail="Credit account not found")

        account.balance_tokens += tokens
        balance_after = account.balance_tokens

        # Record payment
        payment = Payment(
            user_id=user_id,
            provider="stripe",
            provider_payment_id=stripe_session_id,
            amount_cents=amount_cents,
            currency="SGD",
            tokens_purchased=tokens,
            status="paid",
            idempotency_key=stripe_session_id,
        )
        db.add(payment)
        await db.flush()  # populate payment.id

        # Record ledger entry
        ledger = CreditLedger(
            credit_account_id=account.id,
            user_id=user_id,
            entry_type="purchase",
            token_delta=tokens,
            balance_after=balance_after,
            reference_type="payment",
            reference_id=payment.id,
            idempotency_key=f"ledger_{stripe_session_id}",
            note=f"Purchased {tokens} tokens via Stripe",
        )
        db.add(ledger)
        await db.commit()

    return {"ok": True}
