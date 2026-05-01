export default function Privacy() {
  return (
    <main className="legal-shell">
      <div className="container">

        <section className="legal-hero">
          <p className="eyebrow">Privacy Policy</p>
          <h1>How we handle your data</h1>
          <p className="lead">
            DraftProof processes academic documents to provide writing integrity
            signals and review guidance. This page explains what data we collect,
            how we use it, and the choices you have.
          </p>
          <p className="legal-date">Last updated: May 2026</p>
        </section>

        <section className="legal-section">
          <h2>Data We Collect</h2>
          <h3>Account Information</h3>
          <p>
            When you sign in with Google or Microsoft, we receive your name and
            email address from the OAuth provider. We also fetch your profile
            photo when available to personalise your experience.
          </p>
          <h3>Documents</h3>
          <p>
            Files you upload for scanning — typically essays, reports, or
            research papers — are stored temporarily so we can analyse them and
            generate your report.
          </p>
          <h3>Scan Results</h3>
          <p>
            The analysis reports we produce (similarity scores, flagged
            passages, source matches) are stored so you can revisit them from
            your dashboard.
          </p>
          <h3>Payment Information</h3>
          <p>
            We use Stripe to process payments. We never see or store your credit
            card number — only a tokenised confirmation and a record of your
            token purchases.
          </p>
        </section>

        <section className="legal-section">
          <h2>How We Use Your Data</h2>
          <ul>
            <li><strong>Document scanning</strong> — your uploaded files are analysed to produce integrity reports.</li>
            <li><strong>Report delivery</strong> — scan results are saved so you can view and download them.</li>
            <li><strong>Account management</strong> — email and name from OAuth are used to identify your account.</li>
            <li><strong>Token billing</strong> — Stripe processes one-time token purchases; we track your balance.</li>
          </ul>
        </section>

        <section className="legal-section">
          <h2>Data Storage</h2>
          <p>
            Uploaded documents and generated reports are stored in
            <strong> Cloudflare R2</strong> object storage with server-side
            encryption. Our application servers and database are hosted on
            Koyeb's infrastructure with encrypted connections.
          </p>
          <p>
            All data is processed and stored within the regions configured for
            our hosting and storage providers.
          </p>
        </section>

        <section className="legal-section">
          <h2>Third-Party Services</h2>
          <table className="legal-table">
            <thead>
              <tr><th>Service</th><th>Purpose</th><th>Data shared</th></tr>
            </thead>
            <tbody>
              <tr>
                <td>Google / Microsoft</td>
                <td>Sign-in (OAuth 2.0)</td>
                <td>Name, email, profile photo</td>
              </tr>
              <tr>
                <td>Stripe</td>
                <td>Payment processing</td>
                <td>Card details (handled entirely by Stripe)</td>
              </tr>
              <tr>
                <td>OpenAI</td>
                <td>Document analysis &amp; reporting</td>
                <td>Document text for scanning</td>
              </tr>
              <tr>
                <td>Cloudflare R2</td>
                <td>File storage</td>
                <td>Uploaded documents &amp; reports</td>
              </tr>
            </tbody>
          </table>
        </section>

        <section className="legal-section">
          <h2>Data Retention &amp; Deletion</h2>
          <p>
            <strong>You are in control.</strong> You can delete individual
            documents and reports from your dashboard at any time. If you
            request account deletion, we will remove your personal data,
            uploaded files, and scan history.
          </p>
          <p>
            To request deletion, contact us at the email below.
          </p>
        </section>

        <section className="legal-section">
          <h2>Cookies</h2>
          <p>
            DraftProof uses a single <strong>httpOnly session cookie</strong> to
            maintain your authenticated session. We do not use tracking cookies,
            advertising pixels, or third-party analytics scripts.
          </p>
        </section>

        <section className="legal-section">
          <h2>Contact</h2>
          <p>
            Questions about this policy? Reach us at{' '}
            <a href="mailto:support@draftproof.app">support@draftproof.app</a>.
          </p>
        </section>

      </div>
    </main>
  );
}
