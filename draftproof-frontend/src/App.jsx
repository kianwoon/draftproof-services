import { useEffect } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Header from './components/Header';
import Footer from './components/Footer';
import Seo from './components/Seo';
import ProtectedRoute from './components/ProtectedRoute';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import SignIn from './pages/SignIn';
import AuthCallback from './pages/AuthCallback';
import Scan from './pages/Scan';
import Report from './pages/Report';
import Rewrite from './pages/Rewrite';
import Reports from './pages/Reports';
import Pricing from './pages/Pricing';
import FAQ from './pages/FAQ';
import Why from './pages/Why';
import Privacy from './pages/Privacy';
import Security from './pages/Security';
import BuyTokens from './pages/BuyTokens';
import PurchaseHistory from './pages/PurchaseHistory';

function HomeRedirect() {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (loading) return null;
  // Allow hash anchors (e.g. #engine) to show landing page for signed-in users
  if (user && !location.hash) return <Navigate to="/dashboard" replace />;
  return (
    <>
      <Landing />
      <ScrollToHash />
    </>
  );
}

function ScrollToHash() {
  const location = useLocation();
  useEffect(() => {
    if (!location.hash) return undefined;

    const id = location.hash.slice(1);
    const scrollToTarget = () => {
      const el = document.getElementById(id);
      if (el) {
        const shell = el.closest('.snap-shell');
        if (shell) {
          const top = Math.max(0, el.offsetTop - shell.offsetTop);
          shell.scrollTo({ top, behavior: 'auto' });
        } else {
          el.scrollIntoView({ behavior: 'auto' });
        }
      }
    };

    requestAnimationFrame(scrollToTarget);
    const retry = window.setTimeout(scrollToTarget, 100);
    return () => window.clearTimeout(retry);
  }, [location.hash]);
  return null;
}

export default function App() {
  const { pathname } = useLocation();
  const hideFooter = pathname === '/';

  return (
    <AuthProvider>
      <div className="app-shell">
        <Seo />
        <Header />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<HomeRedirect />} />
            <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/pricing" element={<Pricing />} />
            <Route path="/faq" element={<FAQ />} />
            <Route path="/why" element={<Why />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="/security" element={<Security />} />
            <Route path="/buy" element={<ProtectedRoute><BuyTokens /></ProtectedRoute>} />
            <Route path="/history" element={<ProtectedRoute><PurchaseHistory /></ProtectedRoute>} />
            <Route path="/signin" element={<SignIn />} />
            <Route path="/auth/callback" element={<AuthCallback />} />
            <Route path="/scan" element={<ProtectedRoute><Scan /></ProtectedRoute>} />
            <Route path="/reports" element={<ProtectedRoute><Reports /></ProtectedRoute>} />
            <Route path="/report/:id" element={<ProtectedRoute><Report /></ProtectedRoute>} />
            <Route path="/rewrite/:rewriteId" element={<ProtectedRoute><Rewrite /></ProtectedRoute>} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        {!hideFooter && <Footer />}
      </div>
    </AuthProvider>
  );
}
