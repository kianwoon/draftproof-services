const configuredTokenPriceSgd =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_TOKEN_PRICE_SGD)
  || (typeof process !== 'undefined' && process.env?.VITE_TOKEN_PRICE_SGD)
  || '0.90';

export const TOKEN_PRICE_SGD = normalizePrice(configuredTokenPriceSgd);
export const TOKEN_CURRENCY_CODE = 'SGD';
export const TOKEN_CURRENCY_LABEL = 'SGD $';
export const SCAN_TOKENS_PER_1000_WORDS = 1;
export const REWRITE_TOKENS_PER_1000_WORDS = 5;

export function formatSgdAmount(amount) {
  return normalizePrice(amount).toFixed(2);
}

function normalizePrice(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0.90;
}
