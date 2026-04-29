import { Link } from 'react-router-dom';

export default function Header() {
  return (
    <header className="site-header" aria-label="Main navigation">
      <Link to="/" className="brand" aria-label="DraftProof home">
        <span className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 32 32" role="img">
            <path d="M16 3 27 7v8c0 7.2-4.6 11.8-11 14C9.6 26.8 5 22.2 5 15V7l11-4Z" />
            <path d="m10.8 15.9 3.4 3.3 7.4-8" />
          </svg>
        </span>
        <span>DraftProof</span>
      </Link>

      <nav className="nav-links" aria-label="Primary">
        <a href="#product">Product</a>
        <a href="#engine">How it works</a>
        <a href="#report">Sample report</a>
        <a href="#audience">Resources</a>
      </nav>

      <Link to="/scan" className="btn btn-primary btn-small">
        Run a pre-submission check
      </Link>
    </header>
  );
}
