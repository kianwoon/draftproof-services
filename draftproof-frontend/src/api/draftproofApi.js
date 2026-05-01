import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
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

// Reports
export const getReport = (reportId, opts = {}) => api.get(`/reports/${reportId}`, opts);

// Payments
export const getPurchaseHistory = (page = 1, perPage = 5, opts = {}) =>
  api.get('/payments/history', { params: { page, per_page: perPage }, signal: opts.signal });

// Rewrites
export const getSuggestion = (issueId) => api.get(`/rewrites/${issueId}`);
export const applySuggestion = (issueId, suggestionId) =>
  api.post(`/rewrites/${issueId}/apply`, { suggestion_id: suggestionId });

export default api;
