import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      const isAuthCheck = err.config?.url?.endsWith('/auth/me');
      if (!isAuthCheck && !window.location.pathname.startsWith('/signin')) {
        sessionStorage.setItem('auth_next', window.location.pathname);
        window.location.href = '/signin?error=' + encodeURIComponent('Session expired. Please sign in again.');
      }
    }
    return Promise.reject(err);
  },
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
export const startScan = (documentId) => api.post('/scans/', { document_id: documentId });
export const startScanWithText = (text) => api.post('/scans/', { document_id: 'paste', text });
export const getScanStatus = (scanId) => api.get(`/scans/${scanId}`);
export const listScans = () => api.get('/scans/');

// Reports
export const getReport = (reportId) => api.get(`/reports/${reportId}`);

// Rewrites
export const getSuggestion = (issueId) => api.get(`/rewrites/${issueId}`);
export const applySuggestion = (issueId, suggestionId) =>
  api.post(`/rewrites/${issueId}/apply`, { suggestion_id: suggestionId });

export default api;
