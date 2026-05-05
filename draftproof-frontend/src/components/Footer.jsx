import { Link } from 'react-router-dom';
import CodeTexture from './CodeTexture';

export default function Footer() {
  return (
    <div className="global-footer-wrap">
      <CodeTexture id="globalFooter" className="footer-code-texture" />
      <footer className="site-footer">
        <Link to="/" className="brand footer-brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" role="img">
              <path d="M16 3 27 7v8c0 7.2-4.6 11.8-11 14C9.6 26.8 5 22.2 5 15V7l11-4Z" />
              <path d="m10.8 15.9 3.4 3.3 7.4-8" />
            </svg>
          </span>
          <span>DraftProof</span>
        </Link>

        <div className="footer-links">
          <Link to="/#product">Product</Link>
          <Link to="/#engine">How it works</Link>
          <Link to="/#report">Sample report</Link>
          <Link to="/pricing">Pricing</Link>
          <Link to="/privacy">Privacy</Link>
          <Link to="/security">Security</Link>
        </div>

        <p>
          DraftProof provides writing integrity signals and review guidance.
          It does not determine misconduct, plagiarism, or AI authorship.
        </p>
      </footer>
    </div>
  );
}
