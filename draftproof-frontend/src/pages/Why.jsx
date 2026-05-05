import { Link } from 'react-router-dom';
import CodeTexture from '../components/CodeTexture';

export default function Why() {
  return (
    <main className="why-shell">
      <div className="container">

        {/* Hero */}
        <section className="why-hero app-hero app-hero-dark">
          <CodeTexture id="whyHero" />
          <div>
          <p className="eyebrow">Why DraftProof Exists</p>
          <h1>
            We built DraftProof because the way people read, learn, write, and
            cite has changed.
          </h1>
          <p className="lead">
            Students and researchers are no longer working in a clean
            information environment. They are surrounded by AI summaries,
            synthetic explanations, rewritten articles, auto-generated search
            answers, and content that often looks polished before anyone checks
            whether it is grounded.
          </p>
          <p className="lead">
            The problem is no longer only plagiarism. The problem is trust.
          </p>
          <p className="why-punch">The source is getting harder to see.</p>
          </div>
        </section>

        {/* Section 1 */}
        <section className="why-section">
          <span className="why-num">01</span>
          <h2>We now learn from synthetic information</h2>
          <p>
            Today, many of us are surrounded by AI-generated information.
          </p>
          <p>
            Search engines, chatbots, study tools, social platforms, writing
            assistants, and productivity software now summarize knowledge
            before we even reach the original source. In many cases, users read
            the answer first and only look for the source later — if they look
            at all.
          </p>
          <p>That changes how writing happens.</p>
          <p>
            A student may not intentionally copy. A researcher may not
            intentionally misrepresent a source. But when the first layer of
            information is already summarized, simplified, or generated, the
            final writing can become detached from the original evidence.
          </p>
          <p className="why-highlight">
            DraftProof was built to help close that gap.
          </p>
        </section>

        {/* Section 2 */}
        <section className="why-section">
          <span className="why-num">02</span>
          <h2>Traditional media is no longer the only information source</h2>
          <p>
            For decades, most public information came through human-led
            editorial channels: newspapers, journals, books, institutional
            reports, and official publications.
          </p>
          <p>That world has changed.</p>
          <p>
            Today, information moves through automated feeds, AI-assisted
            newsrooms, content farms, social posts, generated summaries, and
            reposted material. Even credible organizations may use AI to draft,
            translate, summarize, or distribute content.
          </p>
          <p>
            This does not automatically make the content false.
          </p>
          <p>
            But it does mean readers and writers need a stronger habit: check
            the source, check the claim, check the context.
          </p>
          <p className="why-punch">Polished does not mean proven.</p>
        </section>

        {/* Section 3 */}
        <section className="why-section">
          <span className="why-num">03</span>
          <h2>AI detection alone is not enough</h2>
          <p>
            A lot of tools try to answer one narrow question:
          </p>
          <p className="why-quote">"Was this written by AI?"</p>
          <p>We think that is the wrong starting point.</p>
          <p>
            AI detection is uncertain. Human writing can look predictable.
            Formal academic writing can look machine-like. Non-native English
            writing can be unfairly flagged. Technical writing often uses
            repeated patterns because precision matters.
          </p>
          <p>
            So DraftProof does not treat an AI score as a verdict.
          </p>
          <p>Instead, we ask better questions:</p>
          <ul className="why-list">
            <li>Is the claim supported?</li>
            <li>Is the citation relevant?</li>
            <li>Is the wording too generic?</li>
            <li>Is the text grounded in the source?</li>
            <li>Does the reader know what needs to be fixed?</li>
          </ul>
          <p className="why-punch">A score is not feedback.</p>
        </section>

        {/* Section 4 */}
        <section className="why-section">
          <span className="why-num">04</span>
          <h2>Plagiarism has also changed</h2>
          <p>
            Plagiarism used to be easier to define: copying words without
            credit.
          </p>
          <p>Now the risks are more subtle.</p>
          <p>
            A paragraph may be paraphrased from an AI summary. A claim may come
            from a source that was never actually read. A citation may be
            attached to a sentence it does not support. A rewrite tool may make
            copied content look original while leaving the underlying idea
            unchanged.
          </p>
          <p>
            That is why DraftProof looks beyond surface similarity.
          </p>
          <p className="why-highlight">
            We focus on writing integrity: citation gaps, source grounding,
            predictable phrasing, rewrite risk, and review-only signals that
            deserve human attention.
          </p>
          <p className="why-punch">
            Original wording is not always original work.
          </p>
        </section>

        {/* Section 5 */}
        <section className="why-section">
          <span className="why-num">05</span>
          <h2>Students need guidance, not fear</h2>
          <p>Most students are not trying to cheat.</p>
          <p>
            Many are trying to survive a confusing writing environment where AI
            tools are everywhere, rules are unclear, and feedback often comes
            too late.
          </p>
          <p>
            DraftProof was built to help before submission, not punish after
            submission.
          </p>
          <p>The goal is to show users what needs attention:</p>
          <ul className="why-list">
            <li>Where to add a citation</li>
            <li>Where to review a source</li>
            <li>Where writing sounds generic</li>
            <li>Where rewriting is useful</li>
            <li>Where no action is needed</li>
          </ul>
          <p className="why-punch">Fix before you submit.</p>
        </section>

        {/* Section 6 — Beliefs */}
        <section className="why-section why-beliefs">
          <span className="why-num">06</span>
          <h2>What DraftProof believes</h2>
          <p>
            We believe writing tools should be fair, transparent, and useful.
          </p>
          <ul className="why-beliefs-list">
            <li>
              We do not believe every AI-like sentence is misconduct.
            </li>
            <li>
              We do not believe every similarity match is plagiarism.
            </li>
            <li>
              We do not believe students should be judged by black-box scores.
            </li>
            <li>
              We do not believe rewriting everything makes writing more honest.
            </li>
          </ul>
          <p className="why-highlight">
            We believe users deserve clear, evidence-based feedback that helps
            them improve their work.
          </p>
        </section>

        {/* Final CTA */}
        <section className="why-cta">
          <h2>
            The world now produces more information than people can easily
            verify.
          </h2>
          <p>
            Before a paper, report, or essay is submitted, users need a way to
            check whether the work is properly grounded, clearly written, and
            responsibly supported.
          </p>
          <p className="why-highlight">DraftProof is that review layer.</p>
          <div className="hero-actions" style={{ justifyContent: 'center' }}>
            <Link to="/scan" className="btn btn-primary">
              Run a pre-submission review
            </Link>
            <Link to="/#engine" className="btn btn-secondary">
              See how DraftProof works
            </Link>
          </div>
        </section>

      </div>
    </main>
  );
}
