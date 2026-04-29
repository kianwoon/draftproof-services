import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
});

export const getMe = () => api.get('/api/auth/me');
export const logout = () => api.post('/api/auth/logout');
export const googleAuthUrl = `${api.defaults.baseURL || ''}/api/auth/google`;
export const microsoftAuthUrl = `${api.defaults.baseURL || ''}/api/auth/microsoft`;

export default api;
