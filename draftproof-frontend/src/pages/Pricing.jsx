import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Pricing() {
  const { user } = useAuth();

  return (
    <main className="pricing-shell">
      <div className="container">
        <section className="pricing-hero">
          <p className="eyebrow">Pricing</p>
          <h1>Pay per scan. Pay per rewrite.</h1>
          <p className="pricing-lead">
            No subscriptions. No hidden fees. Purchase tokens and use them whenever you need.
          </p>
        </section>

        <div className="pricing-grid">
        <div className="pricing-card">
          <div className="pricing-card-header">
            <h2>Scan</h2>
            <div className="pricing-amount">
              <span className="pricing-currency">$</span>
              <span className="pricing-value">1.90</span>
              <span className="pricing-unit">/ scan</span>
            </div>
          </div>

          <ul className="pricing-features">
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span><strong>1 token</strong> per 1,000 words</span>
            </li>
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span><strong>1,000 words included</strong> per token (larger documents use more tokens)</span>
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
              <span>AI phrasing and authorship signals</span>
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

        <div className="pricing-card pricing-card--rewrite">
          <div className="pricing-card-header">
            <h2>Re-Write</h2>
            <div className="pricing-amount">
              <span className="pricing-currency">$</span>
              <span className="pricing-value">3.80</span>
              <span className="pricing-unit">/ rewrite</span>
            </div>
          </div>

          <ul className="pricing-features">
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span><strong>2 tokens</strong> per rewrite</span>
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
            <li>
              <span className="check" aria-hidden="true">&#10003;</span>
              <span>Revised text with improvements highlighted</span>
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
          <h2>Frequently asked questions</h2>
          <div className="faq-item">
            <h3>What counts as a scan?</h3>
            <p>1 token covers up to 1,000 words. Documents over 1,000 words are automatically charged at 1 token per 1,000 words (rounded up). For example, a 2,500-word document costs 3 tokens.</p>
          </div>
          <div className="faq-item">
            <h3>What does a rewrite do?</h3>
            <p>A rewrite rephrases flagged sections of your document to reduce AI-detected content by approximately 20%, while preserving your original meaning.</p>
          </div>
          <div className="faq-item">
            <h3>Do I need a scan before a rewrite?</h3>
            <p>No. You can request a rewrite independently. However, scanning first helps you see exactly where the AI flags are before deciding what to rewrite.</p>
          </div>
          <div className="faq-item">
            <h3>Do tokens expire?</h3>
            <p>No. Your tokens stay in your account until you use them.</p>
          </div>
          <div className="faq-item">
            <h3>How do I submit my text?</h3>
            <p>Paste your text directly into the scan page. File upload coming soon.</p>
          </div>
        </section>
      </div>
    </main>
  );
}
