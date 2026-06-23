import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import { FreshnessStat } from '../components/PageFreshness';

export default function Eula() {
  const { t } = useTranslation();
  const sections = t('legal.eula.sections', { returnObjects: true });

  return (
    <main className="legal-shell">
      <div className="container">
        <section className="legal-hero app-hero app-hero-dark">
          <CodeTexture id="eulaHero" />
          <div>
            <p className="eyebrow">{t('legal.eula.eyebrow')}</p>
            <h1>{t('legal.eula.title')}</h1>
            <p className="lead">{t('legal.eula.lead')}</p>
          </div>
          <FreshnessStat path="/eula" detail={t('legal.eula.stat')} />
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
      {section.paragraphs?.map((paragraph) => <p key={paragraph}>{linkEmail(paragraph)}</p>)}
      {section.bullets && (
        <ul>
          {section.bullets.map((item) => <li key={item}>{item}</li>)}
        </ul>
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
