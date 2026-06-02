import { Link, Navigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../context/AuthContext';
import CodeTexture from '../components/CodeTexture';

export default function Dashboard() {
  const { user, loading, balance } = useAuth();
  const { t } = useTranslation();

  if (loading) {
    return (
      <div className="container" style={{ paddingTop: 'calc(var(--header-h) + 4rem)', textAlign: 'center' }}>
        <p>{t('common.loading')}</p>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/" replace />;
  }

  const initials = user.email?.charAt(0).toUpperCase() || '?';
  const tokenLabel = balance === null ? t('common.checking') : t('common.token', { count: balance });
  const steps = t('dashboard.steps', { returnObjects: true });
  const rewriteSteps = t('dashboard.rewriteSteps', { returnObjects: true });
  const scanSteps = steps.slice(0, 3);
  const reportStep = steps[3];

  return (
    <main className="dash-shell">
      <div className="container">
        <section className="dash-hero">
          <CodeTexture id="dashboardHero" />
          <div className="dash-welcome">
            {user.avatar_url ? (
              <img src={user.avatar_url} alt={user.email} className="dash-avatar dash-avatar-img" />
            ) : (
              <div className="dash-avatar">{initials}</div>
            )}
            <div>
              <p className="eyebrow">{t('dashboard.workspace')}</p>
              <h1>{t('dashboard.welcome')}</h1>
              <p className="dash-email">{user.email}</p>
            </div>
          </div>

          <div className="dash-hero-panel" aria-label={t('dashboard.accountSummary')}>
            <div>
              <span>{t('dashboard.balance')}</span>
              <strong>{tokenLabel}</strong>
            </div>
            <Link to="/buy" className="btn btn-ghost btn-small">{t('dashboard.buyTokens')}</Link>
          </div>
        </section>

        <section className="dash-grid" aria-label={t('dashboard.actions')}>
          <Link to="/scan" className="dash-primary-card">
            <div>
              <span className="brand-pill">{t('dashboard.preSubmission')}</span>
              <h2>{t('dashboard.startTitle')}</h2>
              <p>{t('dashboard.startBody')}</p>
              <p className="dash-free-note">{t('dashboard.freeScanNote')}</p>
            </div>
            <div className="dash-card-footer">
              <span>{t('dashboard.tokenRate')}</span>
              <strong>{t('dashboard.startScan')}</strong>
            </div>
          </Link>

          <div className="dash-side-stack">
            <Link to="/reports" className="dash-small-card">
              <span className="dash-action-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M7 3.8h7.2L18 7.6v12.6H7V3.8Z" />
                  <path d="M14 3.8v4h4M9.5 11h5M9.5 14h5M9.5 17h3" />
                </svg>
              </span>
              <div>
                <h3>{t('dashboard.viewReports')}</h3>
                <p>{t('dashboard.reportsBody')}</p>
              </div>
            </Link>

            <Link to="/history" className="dash-small-card">
              <span className="dash-action-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M12 6v6l3.5 2" />
                  <path d="M20 12a8 8 0 1 1-2.35-5.65" />
                  <path d="M20 4v5h-5" />
                </svg>
              </span>
              <div>
                <h3>{t('dashboard.purchaseHistory')}</h3>
                <p>{t('dashboard.historyBody')}</p>
              </div>
            </Link>
          </div>
        </section>

        <section className="dash-section">
          <div className="dash-section-heading">
            <p className="eyebrow">{t('dashboard.workflow')}</p>
            <h2>{t('dashboard.workflowTitle')}</h2>
          </div>

          <div className="dash-flow-map">
            <div className="dash-flow-panel dash-flow-panel-scan">
              <div className="dash-workflow-panel-heading">
                <span className="brand-pill">{t('dashboard.scanWorkflowLabel')}</span>
                <h3>{t('dashboard.scanWorkflowTitle')}</h3>
              </div>
              <ol className="dash-steps">
                {scanSteps.map((step, index) => (
                  <li className="dash-step" key={step.title}>
                    <span className="step-num">{index + 1}</span>
                    <strong>{step.title}</strong>
                    <p>{step.body}</p>
                  </li>
                ))}
              </ol>
              <Link to="/scan" className="dash-workflow-link">{t('dashboard.startScan')}</Link>
            </div>

            {reportStep && (
              <div className="dash-report-gate">
                <div className="dash-report-node">
                  <div className="dash-report-flow-label dash-report-flow-label-top">
                    {t('dashboard.scanCreatesReport')}
                  </div>
                  <span>{t('dashboard.reportGateLabel')}</span>
                  <strong>{reportStep.title}</strong>
                  <p>{reportStep.body}</p>
                  <div className="dash-report-flow-label dash-report-flow-label-bottom">
                    {t('dashboard.rewriteStartsFromReport')}
                  </div>
                </div>
              </div>
            )}

            <div className="dash-flow-panel dash-flow-panel-rewrite">
              <div className="dash-workflow-panel-heading">
                <span className="brand-pill">{t('dashboard.rewriteWorkflowLabel')}</span>
                <span className="dash-unlock-pill">{t('dashboard.unlockAfterReport')}</span>
                <h3>{t('dashboard.rewriteWorkflowTitle')}</h3>
              </div>
              <ol className="dash-steps">
                {rewriteSteps.map((step, index) => (
                  <li className="dash-step" key={step.title}>
                    <span className="step-num">{index + 1}</span>
                    <strong>{step.title}</strong>
                    <p>{step.body}</p>
                  </li>
                ))}
              </ol>
              <Link to="/reports" className="dash-workflow-link">{t('dashboard.openReports')}</Link>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
