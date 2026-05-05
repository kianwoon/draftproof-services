import CodeTexture from '../components/CodeTexture';

export default function Security() {
  return (
    <main className="legal-shell">
      <div className="container">

        <section className="legal-hero app-hero app-hero-dark">
          <CodeTexture id="securityHero" />
          <div>
          <p className="eyebrow">Security</p>
          <h1>How we protect your work</h1>
          <p className="lead">
            Academic documents deserve strong protection. Here is how
            DraftProof secures your data at every layer.
          </p>
          </div>
          <div className="app-hero-stat">
            <span>Last updated</span>
            <strong>May 2026</strong>
            <small>Encrypted storage</small>
          </div>
        </section>

        <section className="legal-section">
          <h2>Infrastructure Security</h2>
          <ul>
            <li><strong>HTTPS everywhere</strong> — all connections to DraftProof are encrypted with TLS 1.2+.</li>
            <li><strong>Encrypted storage</strong> — documents and reports are stored in Cloudflare R2 with server-side encryption (AES-256).</li>
            <li><strong>Managed hosting</strong> — our application runs on Koyeb's infrastructure, which provides network isolation, automated patching, and DDoS mitigation.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>Authentication</h2>
          <ul>
            <li><strong>OAuth 2.0 sign-in</strong> — we support Google and Microsoft accounts. We never see or store your password.</li>
            <li><strong>httpOnly JWT cookies</strong> — session tokens are stored in httpOnly, Secure, SameSite=Lax cookies, making them inaccessible to JavaScript and resistant to CSRF.</li>
            <li><strong>No password storage</strong> — because we delegate authentication to Google and Microsoft, we have no passwords to leak or compromise.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>Document Protection</h2>
          <ul>
            <li><strong>Encrypted at rest</strong> — all uploaded documents are encrypted in Cloudflare R2 using AES-256.</li>
            <li><strong>Encrypted in transit</strong> — files travel over HTTPS from your browser to our servers and on to storage.</li>
            <li><strong>User-controlled deletion</strong> — you can delete any document or report from your dashboard at any time.</li>
            <li><strong>Access isolation</strong> — each user can only access their own documents and reports.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>Payment Security</h2>
          <ul>
            <li><strong>Stripe handles all card data</strong> — credit card numbers never touch our servers. Stripe is PCI DSS Level 1 certified.</li>
            <li><strong>Webhook signature verification</strong> — payment confirmations from Stripe are cryptographically verified before processing.</li>
            <li><strong>Token-based billing</strong> — we track a prepaid token balance; no recurring charges or stored payment methods on our side.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>Application Security</h2>
          <ul>
            <li><strong>Input validation</strong> — all user input is validated and sanitised on the server before processing.</li>
            <li><strong>CSRF protection</strong> — SameSite cookie policy and state parameters in OAuth flows prevent cross-site request forgery.</li>
            <li><strong>Minimal attack surface</strong> — we use a focused stack (FastAPI + React) with minimal dependencies.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>Responsible Disclosure</h2>
          <p>
            If you discover a security vulnerability in DraftProof, we
            appreciate responsible disclosure. Please email{' '}
            <a href="mailto:security@draftproof.app">security@draftproof.app</a>{' '}
            with details and we will respond promptly.
          </p>
        </section>

      </div>
    </main>
  );
}
