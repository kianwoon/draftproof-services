# Web app — UI kit

A high-fidelity, interactive recreation of the **signed-in DraftProof product**, lifted from the
production React pages (`draftproof-frontend/src/pages/{SignIn,Dashboard,Scan}.jsx`) and the real
stylesheet (`site-master.css`, copied in verbatim).

## What it is
A click-through of the core product loop:

**Sign in → Dashboard → Scan → Report**

- **Sign in** — navy trust panel (code texture + proof pills) beside the white OAuth card
  (Google / Microsoft, official brand-colored logos). Either button signs you in.
- **Dashboard** — navy hero with avatar + token balance; a primary "start a scan" card beside
  two secondary cards; the two-column scan/rewrite workflow explainer.
- **Scan** — paste-a-draft workspace, pre-filled with a sample essay. Live word count + token
  math (free ≤500 words, 1 token per started 1,000 after). "Start scan" runs a simulated progress
  pass and hands off to the report.
- **Report** — four headline metrics, a findings table (claim → evidence issue → severity →
  suggested fix), an action breakdown (citation repair / rewrite priority / review-only / no
  action), and a right rail with the signal profile, primary fix, and pre-submission checklist.

## Run it
Open `index.html`. No build step — React + Babel load from CDN; the `.jsx` files mount in order.
The header "Scan ▾" dropdown and the dashboard cards all navigate.

## Files
| File | Role |
|---|---|
| `index.html` | Screen router (signin / dashboard / scan / report) + small presentation overrides. |
| `site-master.css` | The production stylesheet, verbatim. |
| `CodeTexture.jsx` | The drifting monospace brand motif (shared with marketing). |
| `AppHeader.jsx` | Signed-in header: brand, nav, Scan dropdown, language, token badge, avatar, sign out. |
| `SignInScreen.jsx` | OAuth sign-in (Google / Microsoft). |
| `DashboardScreen.jsx` | Workspace hero, action cards, workflow panels. |
| `ScanScreen.jsx` | Textarea + word/token meter + simulated scan progress. |
| `ReportScreen.jsx` | Completed review: metrics, findings table, action grid, signal rail. |
| `assets/` | Brand mark SVG. |

## Components covered
AppHeader (+ dropdown, token badge, avatar) · SignIn (OAuth buttons, trust panel) · Dashboard
(hero panel, primary/secondary action cards, numbered workflow steps) · Scan (labelled textarea,
pricing note, meta row, progress bar, review-scope rail) · Report (metric strip, findings table,
severity pills, action grid, mini-stats, primary-action callout, check-list) · buttons, pills,
brand mark.

## Faithfulness notes
- Class names + structure mirror production; copy is verbatim or drawn from the live i18n (`en`).
  The report uses the product's real data model (tiers, citation repair, review-only signals).
- It's a **cosmetic recreation**: the scan "runs" on a timer, OAuth buttons just advance the
  router, and there's no real backend — by design.
- **Presentation overrides** (top of `index.html`, commented, not in the source CSS):
  `html/#root height:auto` for natural scrolling; sign-in drops the fixed-header offset; the
  report action rows get a clean label↔value flex layout.
- Best viewed ≥1080px wide (the registered card renders at 1280).
