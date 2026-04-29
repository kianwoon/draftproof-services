-- ============================================================
-- DraftProof MVP Schema — Credit Ledger Model
-- ============================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- -----------------------------------------------------------
-- 1. Users
-- -----------------------------------------------------------
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  email TEXT NOT NULL,
  email_normalized TEXT NOT NULL,
  display_name TEXT,
  avatar_url TEXT,

  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'suspended', 'deleted')),

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (email_normalized)
);

CREATE INDEX idx_users_email_normalized ON users(email_normalized);

-- -----------------------------------------------------------
-- 2. Login Identities
-- -----------------------------------------------------------
CREATE TABLE user_identities (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

  provider TEXT NOT NULL
    CHECK (provider IN ('google', 'microsoft')),

  provider_user_id TEXT NOT NULL,
  provider_email TEXT,
  provider_email_verified BOOLEAN,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_login_at TIMESTAMPTZ,

  UNIQUE (provider, provider_user_id)
);

CREATE INDEX idx_user_identities_user_id ON user_identities(user_id);

-- -----------------------------------------------------------
-- 3. Credit Accounts
-- -----------------------------------------------------------
CREATE TABLE credit_accounts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  user_id UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,

  balance_tokens INTEGER NOT NULL DEFAULT 0,
  reserved_tokens INTEGER NOT NULL DEFAULT 0,

  currency TEXT NOT NULL DEFAULT 'USD',

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  CHECK (balance_tokens >= 0),
  CHECK (reserved_tokens >= 0)
);

-- -----------------------------------------------------------
-- 4. Credit Ledger (source of truth)
-- -----------------------------------------------------------
CREATE TABLE credit_ledger (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  credit_account_id UUID NOT NULL REFERENCES credit_accounts(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

  entry_type TEXT NOT NULL
    CHECK (
      entry_type IN (
        'purchase',
        'scan_debit',
        'rewrite_debit',
        'reservation',
        'reservation_release',
        'refund',
        'admin_adjustment',
        'expiry'
      )
    ),

  token_delta INTEGER NOT NULL,
  balance_after INTEGER NOT NULL,

  reference_type TEXT,
  reference_id UUID,

  idempotency_key TEXT,
  note TEXT,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (idempotency_key)
);

CREATE INDEX idx_credit_ledger_user_id_created_at
  ON credit_ledger(user_id, created_at DESC);

CREATE INDEX idx_credit_ledger_account_created_at
  ON credit_ledger(credit_account_id, created_at DESC);

CREATE INDEX idx_credit_ledger_reference
  ON credit_ledger(reference_type, reference_id);

-- -----------------------------------------------------------
-- 5. Usage Events
-- -----------------------------------------------------------
CREATE TABLE usage_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  credit_account_id UUID NOT NULL REFERENCES credit_accounts(id) ON DELETE CASCADE,

  event_type TEXT NOT NULL
    CHECK (event_type IN ('scan', 'rewrite', 'citation_repair', 'report_download')),

  tokens_charged INTEGER NOT NULL DEFAULT 0,

  document_id UUID,
  job_id UUID,

  word_count INTEGER,
  file_type TEXT,

  status TEXT NOT NULL DEFAULT 'completed'
    CHECK (status IN ('reserved', 'completed', 'failed', 'refunded')),

  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_usage_events_user_id_created_at
  ON usage_events(user_id, created_at DESC);

CREATE INDEX idx_usage_events_job_id ON usage_events(job_id);

-- -----------------------------------------------------------
-- 6. Payments
-- -----------------------------------------------------------
CREATE TABLE payments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

  provider TEXT NOT NULL
    CHECK (provider IN ('stripe', 'manual', 'promo')),

  provider_payment_id TEXT,
  provider_customer_id TEXT,

  amount_cents INTEGER NOT NULL DEFAULT 0,
  currency TEXT NOT NULL DEFAULT 'USD',

  tokens_purchased INTEGER NOT NULL DEFAULT 0,

  status TEXT NOT NULL
    CHECK (status IN ('pending', 'paid', 'failed', 'refunded', 'cancelled')),

  idempotency_key TEXT,

  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (provider, provider_payment_id),
  UNIQUE (idempotency_key)
);

CREATE INDEX idx_payments_user_id_created_at
  ON payments(user_id, created_at DESC);

-- -----------------------------------------------------------
-- 7. Credit Reservations (for async jobs)
-- -----------------------------------------------------------
CREATE TABLE credit_reservations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  credit_account_id UUID NOT NULL REFERENCES credit_accounts(id) ON DELETE CASCADE,

  job_type TEXT NOT NULL
    CHECK (job_type IN ('scan', 'rewrite', 'citation_repair')),

  job_id UUID NOT NULL,

  tokens_reserved INTEGER NOT NULL,

  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'captured', 'released', 'expired')),

  expires_at TIMESTAMPTZ NOT NULL DEFAULT now() + interval '30 minutes',

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (job_type, job_id)
);

CREATE INDEX idx_credit_reservations_user_status
  ON credit_reservations(user_id, status);

-- -----------------------------------------------------------
-- 8. Pricing Plans
-- -----------------------------------------------------------
CREATE TABLE pricing_plans (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  code TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,

  action_type TEXT NOT NULL
    CHECK (action_type IN ('scan', 'rewrite', 'citation_repair')),

  tokens_required INTEGER NOT NULL,
  max_words INTEGER NOT NULL DEFAULT 1000,

  active BOOLEAN NOT NULL DEFAULT true,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Seed pricing
INSERT INTO pricing_plans (code, name, action_type, tokens_required, max_words)
VALUES
  ('scan_1000_words',    'Scan up to 1,000 words',    'scan',    1, 1000),
  ('rewrite_1000_words', 'Rewrite up to 1,000 words', 'rewrite', 2, 1000);
