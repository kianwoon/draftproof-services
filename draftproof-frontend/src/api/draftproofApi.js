import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
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
export const getScanStatus = (scanId) => api.get(`/scans/${scanId}`);

// Reports
export const getReport = (reportId) => api.get(`/reports/${reportId}`);

// Rewrites
export const getSuggestion = (issueId) => api.get(`/rewrites/${issueId}`);
export const applySuggestion = (issueId, suggestionId) =>
  api.post(`/rewrites/${issueId}/apply`, { suggestion_id: suggestionId });

export default api;
