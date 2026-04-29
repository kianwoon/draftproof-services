import { Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import Header from './components/Header';
import ProtectedRoute from './components/ProtectedRoute';
import Landing from './pages/Landing';
import SignIn from './pages/SignIn';
import AuthCallback from './pages/AuthCallback';
import Scan from './pages/Scan';
import Report from './pages/Report';

export default function App() {
  return (
    <AuthProvider>
      <Header />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/auth/callback" element={<AuthCallback />} />
        <Route path="/scan" element={
          <ProtectedRoute><Scan /></ProtectedRoute>
        } />
        <Route path="/report/:id" element={
          <ProtectedRoute><Report /></ProtectedRoute>
        } />
      </Routes>
    </AuthProvider>
  );
}
