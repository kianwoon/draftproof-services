import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import { FreshnessStat } from '../components/PageFreshness';

export default function Security() {
  const { t } = useTranslation();
  const sections = t('legal.security.sections', { returnObjects: true });

  return (
    <main className="legal-shell">
      <div className="container">
        <section className="legal-hero app-hero app-hero-dark">
          <CodeTexture id="securityHero" />
          <div>
            <p className="eyebrow">{t('legal.security.eyebrow')}</p>
            <h1>{t('legal.security.title')}</h1>
            <p className="lead">{t('legal.security.lead')}</p>
          </div>
          <FreshnessStat path="/security" detail={t('legal.security.stat')} />
        </section>

        {sections.map((section) => (
          <section className="legal-section" key={section.title}>
            <h2>{section.title}</h2>
            {section.paragraphs?.map((paragraph) => <p key={paragraph}>{linkEmail(paragraph)}</p>)}
            {section.bullets && (
              <ul>
                {section.bullets.map((item) => <li key={item}>{item}</li>)}
              </ul>
            )}
          </section>
        ))}
      </div>
    </main>
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
