-- 010: Use SGD as the default currency for new credit accounts and payments.
-- Historical payment rows keep their recorded currency.

ALTER TABLE credit_accounts
  ALTER COLUMN currency SET DEFAULT 'SGD';

ALTER TABLE payments
  ALTER COLUMN currency SET DEFAULT 'SGD';
