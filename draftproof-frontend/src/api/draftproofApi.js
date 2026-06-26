import axios from 'axios';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
  timeout: 300000, // 5 min — rewrite pipeline can take 3+ min
});

// --- Centralized session-expiry handling ----------------------------------
// ProtectedRoute trusts the in-memory `user` (fetched once at app mount); the
// auth cookie can expire while that state is still set, so a page renders fine
// but its first data call comes back 401 "Not authenticated". Without this
// interceptor each page reinvents the handling — and several (Reports, Report's
// initial load, PurchaseHistory, Rewrite) fall through to ErrorReload's 10s
// reload countdown instead of going to sign-in. Here we catch every 401 (and a
// 403 whose detail is "not authenticated") in ONE place and hand it to a
// handler registered by AuthContext, which clears auth state and redirects to
// /signin immediately. Only the authenticated `api` instance carries this;
// authApi (getMe/logout) deliberately does NOT, since getMe legitimately 401s
// for anonymous visitors on every public page and would otherwise loop.
let sessionExpiredHandler = null;
let handlingSessionExpiry = false;

export function registerSessionExpiredHandler(fn) {
  sessionExpiredHandler = fn;
}

export function isAuthExpiryError(error) {
  // Aborted/canceled requests carry no auth meaning — never treat them as expiry.
  if (error?.code === 'ERR_CANCELED' || error?.name === 'CanceledError') return false;
  const status = error?.response?.status;
  if (status === 401) return true;
  const detail = String(error?.response?.data?.detail || '').toLowerCase();
  return status === 403 && detail.includes('not authenticated');
}

api.interceptors.response.use(
  (response) => {
    // A successful authenticated call means we're signed in again — re-arm the
    // guard so a future expiry (e.g. after re-login) is handled.
    handlingSessionExpiry = false;
    return response;
  },
  (error) => {
    if (isAuthExpiryError(error) && sessionExpiredHandler && !handlingSessionExpiry) {
      // Guard so parallel in-flight 401s trigger exactly one redirect.
      handlingSessionExpiry = true;
      sessionExpiredHandler();
    }
    return Promise.reject(error);
  }
);

// Documents
export const uploadDocument = (formData) =>
  api.post('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

export const uploadText = (text) =>
  api.post('/documents/text', { text });

export const getDocument = (id) => api.get(`/documents/${id}`);

// Scans
export const startScan = (documentId, opts = {}) =>
  api.post('/scans/', { document_id: documentId }, opts);
export const startScanWithText = (text, opts = {}) =>
  api.post('/scans/', { document_id: 'paste', text }, opts);
export const getScanStatus = (scanId, opts = {}) => api.get(`/scans/${scanId}`, opts);
export const listScans = (page = 1, perPage = 10, opts = {}) =>
  api.get('/scans/', { params: { page, per_page: perPage }, signal: opts.signal });
export const deleteScan = (scanId) => api.delete(`/scans/${scanId}`);

// Reports
export const getReport = (reportId, opts = {}) => api.get(`/reports/${reportId}`, opts);

// Translation (ESL writing aid in the submitted-content editor)
export const translateText = (text, { source = 'auto', target = 'en' } = {}) =>
  api.post('/translate', { text, source, target });

// Feedback — files a GitHub issue server-side, gated by a Turnstile token.
// payload: { type: 'bug'|'feature', title, body, email?, page_url?, turnstile_token }
export const submitFeedback = (payload) => api.post('/feedback', payload);
// Public runtime config (Turnstile site key) — served by the API so the key
// stays a plain env var instead of a Vite build-time inline.
export const getFeedbackConfig = () => api.get('/feedback/config');

// API keys (for the Google Docs / MS Word add-ins). createApiKey returns the
// clear-text key exactly once, in response.data.key.
export const listApiKeys = () => api.get('/keys/');
export const createApiKey = (name) => api.post('/keys/', { name });
export const revokeApiKey = (keyId) => api.delete(`/keys/${keyId}`);

// Payments
export const getPurchaseHistory = (page = 1, perPage = 5, opts = {}) =>
  api.get('/payments/history', { params: { page, per_page: perPage }, signal: opts.signal });

// Rewrites
export const createRewrite = (scanId) => api.post('/rewrites/', { scan_id: scanId });
export const cancelRewrite = (rewriteId) => api.post(`/rewrites/${rewriteId}/cancel`);
export const getRewriteStatus = (rewriteId) => api.get(`/rewrites/${rewriteId}`);
export const getRewriteReport = (rewriteId) => api.get(`/rewrites/${rewriteId}/report`);
export const regenerateRewriteReport = (rewriteId) => api.post(`/rewrites/${rewriteId}/report/regenerate`);
export const getRewriteDownload = (rewriteId, format) => api.get(`/rewrites/${rewriteId}/download/${format}`);
export const getDetectJson = (rewriteId) => api.get(`/rewrites/${rewriteId}/detect-json`);

export function buildApiEventUrl(path) {
  if (/^https?:\/\//i.test(API_BASE_URL)) {
    const base = new URL(API_BASE_URL);
    return `${base.origin}${base.pathname.replace(/\/$/, '')}${path}`;
  }
  return `${API_BASE_URL.replace(/\/$/, '')}${path}`;
}

export default api;
