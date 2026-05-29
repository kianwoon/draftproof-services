// SignInScreen — the OAuth-only sign-in. Left: navy trust panel with the code
// texture + proof pills. Right: the white card with Google / Microsoft buttons
// (official brand-colored inline SVG logos). Copy verbatim from production.
function SignInScreen({ go }) {
  return (
    <main className="app-page signin-shell">
      <div className="container signin-layout">
        <section className="signin-trust-panel">
          <CodeTexture id="signinTrust" />
          <p className="brand-pill">Trusted pre-submission review</p>
          <h1>Review your draft before it becomes a submission.</h1>
          <p>Sign in to scan for citation gaps, weak source grounding, similarity risk, and
            AI-like writing signals — then revise with a clear plan.</p>
          <div className="signin-proof-list">
            <span>No bypass claims</span>
            <span>PDF report included</span>
            <span>Signals, not verdicts</span>
          </div>
        </section>

        <section className="signin-card" aria-labelledby="signin-title">
          <p className="eyebrow">Secure access</p>
          <h2 id="signin-title">Continue to DraftProof</h2>
          <p>Use your school or work account. We never see your password.</p>

          <div className="signin-buttons">
            <a href="#" className="btn btn-signin btn-google"
              onClick={(e) => { e.preventDefault(); go('dashboard'); }}>
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Continue with Google
            </a>

            <a href="#" className="btn btn-signin btn-microsoft"
              onClick={(e) => { e.preventDefault(); go('dashboard'); }}>
              <svg viewBox="0 0 24 24" width="20" height="20">
                <rect x="1" y="1" width="10" height="10" fill="#F25022"/>
                <rect x="13" y="1" width="10" height="10" fill="#7FBA00"/>
                <rect x="1" y="13" width="10" height="10" fill="#00A4EF"/>
                <rect x="13" y="13" width="10" height="10" fill="#FFB900"/>
              </svg>
              Continue with Microsoft
            </a>
          </div>

          <p className="signin-note">
            By continuing you agree to the Terms and Privacy Policy. DraftProof provides writing
            integrity signals and review guidance — not a misconduct or authorship verdict.
          </p>
        </section>
      </div>
    </main>
  );
}
