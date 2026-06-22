import { useEffect } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { AuthProvider, useAuth } from './context/AuthContext';
import Header from './components/Header';
import AnnouncementBanner from './components/AnnouncementBanner';
import Footer from './components/Footer';
import FeedbackWidget from './components/FeedbackWidget';
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
import Features from './pages/Features';
import Privacy from './pages/Privacy';
import Terms from './pages/Terms';
import Support from './pages/Support';
import Security from './pages/Security';
import EssayChecker from './pages/EssayChecker';
import NotFound from './pages/NotFound';
import BuyTokens from './pages/BuyTokens';
import PurchaseHistory from './pages/PurchaseHistory';
import ApiKeys from './pages/ApiKeys';
import { getLocaleFromPathname, isLocalizablePublicPath } from './localeRouting';

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

function ScrollToRouteTop() {
  const location = useLocation();

  useEffect(() => {
    if (location.hash) return undefined;

    const frame = requestAnimationFrame(() => {
      window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
    });
    return () => cancelAnimationFrame(frame);
  }, [location.hash, location.pathname, location.search]);

  return null;
}

export default function App() {
  const { pathname } = useLocation();
  const { i18n } = useTranslation();
  const hideFooter = pathname === '/' || pathname === '/zh';

  useEffect(() => {
    if (!isLocalizablePublicPath(pathname)) return;

    const routeLocale = getLocaleFromPathname(pathname);
    if (routeLocale !== i18n.resolvedLanguage && routeLocale !== i18n.language) {
      i18n.changeLanguage(routeLocale);
    }
  }, [pathname, i18n]);

  return (
    <AuthProvider>
      <div className="app-shell">
        <Seo />
        <Header />
        <AnnouncementBanner />
        <ScrollToRouteTop />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<HomeRedirect />} />
            <Route path="/zh" element={<HomeRedirect />} />
            <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/pricing" element={<Pricing />} />
            <Route path="/zh/pricing" element={<Pricing />} />
            <Route path="/faq" element={<FAQ />} />
            <Route path="/zh/faq" element={<FAQ />} />
            <Route path="/why" element={<Why />} />
            <Route path="/zh/why" element={<Why />} />
            <Route path="/features" element={<Features />} />
            <Route path="/zh/features" element={<Features />} />
            <Route path="/content-checker" element={<EssayChecker />} />
            <Route path="/zh/content-checker" element={<EssayChecker />} />
            <Route path="/essay-checker" element={<Navigate to="/content-checker" replace />} />
            <Route path="/zh/essay-checker" element={<Navigate to="/zh/content-checker" replace />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="/zh/privacy" element={<Privacy />} />
            <Route path="/terms" element={<Terms />} />
            <Route path="/zh/terms" element={<Terms />} />
            <Route path="/support" element={<Support />} />
            <Route path="/zh/support" element={<Support />} />
            <Route path="/security" element={<Security />} />
            <Route path="/zh/security" element={<Security />} />
            <Route path="/buy" element={<ProtectedRoute><BuyTokens /></ProtectedRoute>} />
            <Route path="/history" element={<ProtectedRoute><PurchaseHistory /></ProtectedRoute>} />
            <Route path="/api-keys" element={<ProtectedRoute><ApiKeys /></ProtectedRoute>} />
            <Route path="/signin" element={<SignIn />} />
            <Route path="/zh/signin" element={<SignIn />} />
            <Route path="/auth/callback" element={<AuthCallback />} />
            <Route path="/scan" element={<ProtectedRoute><Scan /></ProtectedRoute>} />
            <Route path="/reports" element={<ProtectedRoute><Reports /></ProtectedRoute>} />
            <Route path="/report/:id" element={<ProtectedRoute><Report /></ProtectedRoute>} />
            <Route path="/rewrite/:rewriteId" element={<ProtectedRoute><Rewrite /></ProtectedRoute>} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
        {!hideFooter && <Footer />}
        <FeedbackWidget />
      </div>
    </AuthProvider>
  );
}
