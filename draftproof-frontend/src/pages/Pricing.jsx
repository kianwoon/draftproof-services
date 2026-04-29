import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Pricing() {
  const { user } = useAuth();

  return (
    <main className="pricing-shell">
      <div className="container">
        <section className="pricing-hero">
          <p className="eyebrow">Pricing</p>
          <h1>Simple, pay-per-scan pricing.</h1>
          <p className="pricing-lead">
            No subscriptions. No hidden fees. Buy tokens and use them whenever you need.
          </p>
        </section>

        <div className="pricing-grid">
        <div className="pricing-card">
          <div className="pricing-card-header">
            <h2>Per-Scan Token</h2>
            <div className="pricing-amount">
              <span className="pricing-currency">$</span>
              <span className="pricing-value">2.90</span>
              <span className="pricing-unit">/ scan</span>
            </div>
          </div>

          <ul className="pricing-features">
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>1 token per document scan</span>
            </li>
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>Up to <strong>1,000 words</strong> per scan</span>
            </li>
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>Citation gap analysis</span>
            </li>
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>Source integrity review</span>
            </li>
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>Phrasing and authorship signals</span>
            </li>
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>Suggested rewrites</span>
            </li>
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>Downloadable PDF report</span>
            </li>
          </ul>

          <Link
            to={user ? '/scan' : '/signin'}
            className="btn btn-primary pricing-cta"
          >
            {user ? 'Start a scan' : 'Sign in to get started'}
          </Link>
        </div>

        <div className="pricing-card pricing-card--rewrite pricing-card--right">
          <div className="pricing-card-header">
            <h2>Re-Write</h2>
            <div className="pricing-amount">
              <span className="pricing-value pricing-value--sm">2</span>
              <span className="pricing-unit">tokens / rewrite</span>
            </div>
          </div>

          <ul className="pricing-features">
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>Requires <strong>2 tokens</strong> per rewrite</span>
            </li>
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>Up to <strong>1,000 words</strong> per rewrite</span>
            </li>
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>Reduce AI footprint by <strong>20%</strong></span>
            </li>
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>Human-like phrasing improvements</span>
            </li>
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>Preserves original meaning and structure</span>
            </li>
          </ul>

          <Link
            to={user ? '/scan' : '/signin'}
            className="btn btn-primary pricing-cta"
          >
            {user ? 'Start a rewrite' : 'Sign in to get started'}
          </Link>
        </div>
        </div>

        <section className="pricing-faq">
          <h2>Questions</h2>
          <div className="faq-item">
            <h3>What counts as a scan?</h3>
            <p>One scan covers one document up to 1,000 words. If your document exceeds 1,000 words, you can split it into multiple scans.</p>
          </div>
          <div className="faq-item">
            <h3>Do tokens expire?</h3>
            <p>No. Your tokens stay in your account until you use them.</p>
          </div>
          <div className="faq-item">
            <h3>What file formats are supported?</h3>
            <p>We support .docx, .pdf, and .txt files up to 10 MB.</p>
          </div>
        </section>
      </div>
    </main>
  );
}
