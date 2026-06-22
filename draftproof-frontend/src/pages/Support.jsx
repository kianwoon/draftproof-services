import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';

export default function Support() {
  const { t } = useTranslation();
  const sections = t('legal.support.sections', { returnObjects: true });

  return (
    <main className="legal-shell">
      <div className="container">
        <section className="legal-hero app-hero app-hero-dark">
          <CodeTexture id="supportHero" />
          <div>
            <p className="eyebrow">{t('legal.support.eyebrow')}</p>
            <h1>{t('legal.support.title')}</h1>
            <p className="lead">{t('legal.support.lead')}</p>
          </div>
        </section>

        {sections.map((section) => (
          <section className="legal-section" key={section.title}>
            <h2>{section.title}</h2>
            {section.paragraphs?.map((paragraph) => <p key={paragraph}>{linkEmail(paragraph)}</p>)}
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
