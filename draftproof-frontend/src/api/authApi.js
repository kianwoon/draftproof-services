import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
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

export const getMe = () => api.get('/api/auth/me');
export const logout = () => api.post('/api/auth/logout');
export const googleAuthUrl = `${api.defaults.baseURL || ''}/api/auth/google`;
export const microsoftAuthUrl = `${api.defaults.baseURL || ''}/api/auth/microsoft`;

export default api;
