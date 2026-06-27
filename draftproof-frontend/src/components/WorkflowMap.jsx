import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

// The scan → report → rewrite workflow map (moved from the former Dashboard).
// Rendered below the scan form as onboarding context; reveals on scroll-into-view.
export default function WorkflowMap() {
  const { t } = useTranslation();
  const flowRef = useRef(null);
  const [flowVisible, setFlowVisible] = useState(false);

  useEffect(() => {
    const node = flowRef.current;
    if (!node || flowVisible) return;
    if (typeof IntersectionObserver === 'undefined') {
      setFlowVisible(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setFlowVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.25 },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [flowVisible]);

  const steps = t('dashboard.steps', { returnObjects: true });
  const rewriteSteps = t('dashboard.rewriteSteps', { returnObjects: true });
  const manualSteps = t('dashboard.manualSteps', { returnObjects: true });
  const scanSteps = steps.slice(0, 3);
  const reportStep = steps[3];

  return (
    <section className="dash-section">
      <div className="dash-section-heading">
        <p className="eyebrow">{t('dashboard.workflow')}</p>
        <h2>{t('dashboard.workflowTitle')}</h2>
      </div>

      <div
        ref={flowRef}
        className={`dash-flow-map dash-anim${flowVisible ? ' is-animating' : ''}`}
      >
        <div className="dash-flow-panel dash-flow-panel-scan">
          <div className="dash-workflow-panel-heading">
            <span className="brand-pill">{t('dashboard.scanWorkflowLabel')}</span>
            <h3>{t('dashboard.scanWorkflowTitle')}</h3>
          </div>
          <ol className="dash-steps">
            {scanSteps.map((step, index) => (
              <li className="dash-step" key={step.title} style={{ '--i': index }}>
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
            <div className="dash-report-node" style={{ '--i': 3 }}>
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
            <span className="dash-unlock-pill" style={{ '--i': 3 }}>{t('dashboard.unlockAfterReport')}</span>
            <h3>{t('dashboard.rewriteWorkflowTitle')}</h3>
          </div>
          <ol className="dash-steps">
            {rewriteSteps.map((step, index) => (
              <li className="dash-step" key={step.title} style={{ '--i': index + 4 }}>
                <span className="step-num">{index + 1}</span>
                <strong>{step.title}</strong>
                <p>{step.body}</p>
              </li>
            ))}
          </ol>
          <Link to="/reports" className="dash-workflow-link">{t('dashboard.openReports')}</Link>

          <div className="dash-flow-or-divider"><span>or</span></div>

          <div className="dash-workflow-panel-heading">
            <span className="brand-pill brand-pill-muted">{t('dashboard.manualWorkflowLabel')}</span>
            <h3>{t('dashboard.manualWorkflowTitle')}</h3>
          </div>
          <ol className="dash-steps">
            {manualSteps.map((step, index) => (
              <li className="dash-step dash-step-manual" key={step.title} style={{ '--i': index + 8 }}>
                <span className="step-num">{index + 1}</span>
                <strong>{step.title}</strong>
                <p>{step.body}</p>
              </li>
            ))}
          </ol>
          <Link to="/scan" className="dash-workflow-link">{t('dashboard.startScan')}</Link>
        </div>
      </div>
    </section>
  );
}
