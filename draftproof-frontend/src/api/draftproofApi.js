import axios from 'axios';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
  timeout: 300000, // 5 min — rewrite pipeline can take 3+ min
});

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
export const getFreeScanUsage = () => api.get('/scans/free-usage');

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
