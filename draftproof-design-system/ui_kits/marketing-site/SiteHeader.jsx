// BrandMark — the one true DraftProof icon: shield outline + interior check.
// Inherits color from context (navy on light, green on dark via .site-header-dark).
function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 32 32" role="img">
        <path d="M16 3 27 7v8c0 7.2-4.6 11.8-11 14C9.6 26.8 5 22.2 5 15V7l11-4Z" />
        <path d="m10.8 15.9 3.4 3.3 7.4-8" />
      </svg>
    </span>
  );
}

// SiteHeader — fixed, blurred. Dark variant (site-header-dark) is used on the
// landing page so the brand + nav sit on the navy hero. Nav links are the real
// marketing IA. The chevron "Scan" dropdown only appears when signed in (app),
// so the marketing header just shows the public links + the green CTA.
function SiteHeader({ onNav }) {
  const links = [
    ['Why', '#product'],
    ['Essay checker', '#report'],
    ['Pricing', '#pricing'],
    ['FAQ', '#faq'],
    ['Sample report', '#report'],
  ];
  return (
    <header className="site-header site-header-dark" aria-label="Main navigation">
      <a href="#hero" className="brand" aria-label="DraftProof home" onClick={onNav}>
        <BrandMark />
        <span>DraftProof</span>
      </a>

      <nav className="nav-links" aria-label="Primary">
        {links.map(([label, href]) => (
          <a key={label} href={href} onClick={onNav}>{label}</a>
        ))}
      </nav>

      <div className="header-actions">
        <span className="language-switcher language-switcher-compact" aria-hidden="true">
          <span>Language</span>
          <select defaultValue="en" aria-label="Language">
            <option value="en">EN</option>
            <option value="zh">中文</option>
          </select>
        </span>
        <a href="#" className="btn btn-primary btn-small" onClick={onNav}>Start review</a>
      </div>
    </header>
  );
}
