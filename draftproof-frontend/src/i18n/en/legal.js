export const legal = {
  "lastUpdated": "Last updated",
  "may2026": "May 2026",
  "privacy": {
    "eyebrow": "Privacy Policy",
    "title": "How we handle your data",
    "lead": "DraftProof processes academic documents to provide writing integrity signals and review guidance. This page explains what data we collect, how we use it, and the choices you have.",
    "stat": "User-controlled deletion",
    "sections": [
      {
        "title": "Data We Collect",
        "items": [
          {
            "title": "Account Information",
            "body": "When you sign in with Google or Microsoft, we receive your name and email address from the OAuth provider. We also fetch your profile photo when available to personalise your experience."
          },
          {
            "title": "Documents",
            "body": "Files you upload for scanning, typically essays, reports, or research papers, are stored temporarily so we can analyse them and generate your report."
          },
          {
            "title": "Scan Results",
            "body": "The analysis reports we produce are stored so you can revisit them from your dashboard."
          },
          {
            "title": "Payment Information",
            "body": "We use Stripe to process payments. We never see or store your credit card number, only a tokenised confirmation and a record of your token purchases."
          }
        ]
      },
      {
        "title": "How We Use Your Data",
        "bullets": [
          "Document scanning: uploaded files are analysed to produce integrity reports.",
          "Report delivery: scan results are saved so you can view and download them.",
          "Account management: email and name from OAuth are used to identify your account.",
          "Token billing: Stripe processes one-time token purchases; we track your balance."
        ]
      },
      {
        "title": "Data Storage",
        "paragraphs": [
          "Uploaded documents and generated reports are stored temporarily in Cloudflare R2 object storage with server-side encryption. Generated scan and rewrite reports stay available in DraftProof for 3 days before the system copy is purged.",
          "Our application servers and database are hosted on Koyeb infrastructure with encrypted connections. All data is processed and stored within the regions configured for our hosting and storage providers."
        ]
      },
      {
        "title": "Third-Party Services",
        "table": {
          "headers": [
            "Service",
            "Purpose",
            "Data shared"
          ],
          "rows": [
            [
              "Google / Microsoft",
              "Sign-in (OAuth 2.0)",
              "Name, email, profile photo"
            ],
            [
              "Stripe",
              "Payment processing",
              "Card details handled entirely by Stripe"
            ],
            [
              "Cloudflare R2",
              "File storage",
              "Uploaded documents and reports"
            ]
          ]
        }
      },
      {
        "title": "Data Retention & Deletion",
        "paragraphs": [
          "Generated scan and rewrite reports stay available in DraftProof for 3 days, then are purged from the system. Report copies are sent to your mailbox when email delivery is enabled so you can keep your own record.",
          "You are in control. You can delete individual documents and reports from your dashboard at any time. If you request account deletion, we will remove your personal data, uploaded files, and scan history.",
          "To request deletion, contact us at the email below."
        ]
      },
      {
        "title": "Cookies",
        "paragraphs": [
          "DraftProof uses a single httpOnly session cookie to maintain your authenticated session. We do not use tracking cookies, advertising pixels, or third-party analytics scripts."
        ]
      },
      {
        "title": "Contact",
        "paragraphs": [
          "Questions about this policy? Reach us at support@draftproof.app."
        ]
      }
    ]
  },
  "security": {
    "eyebrow": "Security",
    "title": "How we protect your work",
    "lead": "Academic documents deserve strong protection. Here is how DraftProof secures your data at every layer.",
    "stat": "Encrypted storage",
    "sections": [
      {
        "title": "Infrastructure Security",
        "bullets": [
          "HTTPS everywhere: all connections to DraftProof are encrypted with TLS 1.2+.",
          "Encrypted storage: documents and reports are stored in Cloudflare R2 with server-side encryption (AES-256).",
          "Managed hosting: our application runs on Koyeb infrastructure with network isolation, automated patching, and DDoS mitigation."
        ]
      },
      {
        "title": "Authentication",
        "bullets": [
          "OAuth 2.0 sign-in: we support Google and Microsoft accounts. We never see or store your password.",
          "httpOnly JWT cookies: session tokens are stored in httpOnly, Secure, SameSite=Lax cookies.",
          "No password storage: authentication is delegated to Google and Microsoft."
        ]
      },
      {
        "title": "Document Protection",
        "bullets": [
          "Encrypted at rest: all uploaded documents are encrypted in Cloudflare R2 using AES-256.",
          "Encrypted in transit: files travel over HTTPS from your browser to our servers and on to storage.",
          "User-controlled deletion: you can delete any document or report from your dashboard at any time.",
          "Access isolation: each user can only access their own documents and reports."
        ]
      },
      {
        "title": "Payment Security",
        "bullets": [
          "Stripe handles all card data: credit card numbers never touch our servers. Stripe is PCI DSS Level 1 certified.",
          "Webhook signature verification: payment confirmations from Stripe are cryptographically verified before processing.",
          "Token-based billing: we track a prepaid token balance; no recurring charges or stored payment methods on our side."
        ]
      },
      {
        "title": "Application Security",
        "bullets": [
          "Input validation: all user input is validated and sanitised on the server before processing.",
          "CSRF protection: SameSite cookie policy and state parameters in OAuth flows prevent cross-site request forgery.",
          "Minimal attack surface: we use a focused stack (FastAPI + React) with minimal dependencies."
        ]
      },
      {
        "title": "Responsible Disclosure",
        "paragraphs": [
          "If you discover a security vulnerability in DraftProof, we appreciate responsible disclosure. Please email security@draftproof.app with details and we will respond promptly."
        ]
      }
    ]
  }
};
