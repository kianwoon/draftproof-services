-- Personal API keys — programmatic auth for the Google Docs / MS Word add-ins.
--
-- A user mints a key from the web dashboard and pastes it into the extension.
-- The extension authenticates each scan with the key (Authorization: Bearer or
-- X-API-Key header), which resolves to the owning user and bills their existing
-- credit balance — no new currency, no new cost path.
--
-- Security model:
--   * Only the SHA-256 hex digest of the key is stored. Keys are high-entropy
--     random tokens, so a fast hash is correct (bcrypt is for passwords). The
--     clear-text key is shown exactly once, at creation.
--   * key_prefix holds the display-safe leading chars (e.g. "dp_live_a1b2c3d4")
--     so the dashboard can list keys without exposing the secret.
--   * revoked_at NULL == active; revocation is a soft delete that preserves the
--     row for audit and prevents hash reuse.

CREATE TABLE IF NOT EXISTS api_keys (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name         TEXT NOT NULL DEFAULT 'API key',
    key_prefix   TEXT NOT NULL,
    key_hash     TEXT NOT NULL UNIQUE,
    last_used_at TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
-- Auth path looks up by key_hash filtered to active keys; the UNIQUE constraint
-- already indexes key_hash, this partial index keeps the active-key scan tight.
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(key_hash) WHERE revoked_at IS NULL;
