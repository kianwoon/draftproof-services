const configuredTokenPriceUsd =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_TOKEN_PRICE_USD)
  || (typeof process !== 'undefined' && process.env?.VITE_TOKEN_PRICE_USD)
  || '0.90';

export const TOKEN_PRICE_USD = normalizePrice(configuredTokenPriceUsd);
export const SCAN_TOKENS_PER_1000_WORDS = 1;
export const REWRITE_TOKENS_PER_1000_WORDS = 5;

export function formatUsdAmount(amount) {
  return normalizePrice(amount).toFixed(2);
}

function normalizePrice(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0.90;
}
