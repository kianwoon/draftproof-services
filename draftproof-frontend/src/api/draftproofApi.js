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
export const startScan = (documentId) => api.post('/scans/', { document_id: documentId });
export const startScanWithText = (text) => api.post('/scans/', { document_id: 'paste', text });
export const getScanStatus = (scanId, opts = {}) => api.get(`/scans/${scanId}`, opts);
export const listScans = (page = 1, perPage = 10, opts = {}) =>
  api.get('/scans/', { params: { page, per_page: perPage }, signal: opts.signal });
export const deleteScan = (scanId) => api.delete(`/scans/${scanId}`);

// Reports
export const getReport = (reportId, opts = {}) => api.get(`/reports/${reportId}`, opts);

// Payments
export const getPurchaseHistory = (page = 1, perPage = 5, opts = {}) =>
  api.get('/payments/history', { params: { page, per_page: perPage }, signal: opts.signal });

// Rewrites
export const createRewrite = (scanId) => api.post('/rewrites/', { scan_id: scanId });
export const getRewriteStatus = (rewriteId) => api.get(`/rewrites/${rewriteId}`);
export const getRewriteReport = (rewriteId) => api.get(`/rewrites/${rewriteId}/report`);
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
