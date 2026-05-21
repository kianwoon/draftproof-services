import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import { FreshnessStat } from '../components/PageFreshness';

export default function Privacy() {
  const { t } = useTranslation();
  const sections = t('legal.privacy.sections', { returnObjects: true });

  return (
    <main className="legal-shell">
      <div className="container">
        <section className="legal-hero app-hero app-hero-dark">
          <CodeTexture id="privacyHero" />
          <div>
            <p className="eyebrow">{t('legal.privacy.eyebrow')}</p>
            <h1>{t('legal.privacy.title')}</h1>
            <p className="lead">{t('legal.privacy.lead')}</p>
          </div>
          <FreshnessStat path="/privacy" detail={t('legal.privacy.stat')} />
        </section>

        {sections.map((section) => (
          <LegalSection key={section.title} section={section} />
        ))}
      </div>
    </main>
  );
}

function LegalSection({ section }) {
  return (
    <section className="legal-section">
      <h2>{section.title}</h2>
      {section.items?.map((item) => (
        <div key={item.title}>
          <h3>{item.title}</h3>
          <p>{item.body}</p>
        </div>
      ))}
      {section.paragraphs?.map((paragraph) => <p key={paragraph}>{linkEmail(paragraph)}</p>)}
      {section.bullets && (
        <ul>
          {section.bullets.map((item) => <li key={item}>{item}</li>)}
        </ul>
      )}
      {section.table && (
        <table className="legal-table">
          <thead>
            <tr>{section.table.headers.map((header) => <th key={header}>{header}</th>)}</tr>
          </thead>
          <tbody>
            {section.table.rows.map((row) => (
              <tr key={row.join('|')}>{row.map((cell) => <td key={cell}>{cell}</td>)}</tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function linkEmail(text) {
  const email = text.match(/[\w.-]+@draftproof\.app/)?.[0];
  if (!email) return text;
  const [before, after] = text.split(email);
  return (
    <>
      {before}
      <a href={`mailto:${email}`}>{email}</a>
      {after}
    </>
  );
}
