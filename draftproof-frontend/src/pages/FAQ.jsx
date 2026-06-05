import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

export default function FAQ() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);
  const groups = t('faqPage.groups', { returnObjects: true });
  const totalQuestions = groups.reduce((count, group) => count + group.items.length, 0);
  const featuredGroups = groups.slice(0, 3);

  return (
    <main className="faq-shell">
      <div className="container">
        <section className="faq-hero app-hero app-hero-dark">
          <CodeTexture id="faqHero" />
          <div>
            <p className="eyebrow">{t('faqPage.eyebrow')}</p>
            <h1>{t('faqPage.title')}</h1>
            <p className="lead">{t('faqPage.lead')}</p>
          </div>
          <div className="app-hero-stat">
            <span>{t('faqPage.heroStatLabel')}</span>
            <strong>{t('faqPage.heroStatValue')}</strong>
            <small>{t('faqPage.heroStatDetail')}</small>
          </div>
        </section>

        <section className="faq-overview" aria-label={t('faqPage.overviewLabel')}>
          <article>
            <span>{t('faqPage.questionCountLabel')}</span>
            <strong>{totalQuestions}</strong>
            <p>{t('faqPage.questionCountDetail')}</p>
          </article>
          {featuredGroups.map((group) => (
            <a href={`#${group.id}`} key={group.id}>
              <span>{group.kicker}</span>
              <strong>{group.title}</strong>
            </a>
          ))}
        </section>

        <section className="faq-intro">
          <div>
            <h2>{t('faqPage.helpTitle')}</h2>
            <p>{t('faqPage.helpBody')}</p>
          </div>
          <div className="faq-intro-actions">
            <Link to="/signin?next=/scan" className="btn btn-primary">{t('faqPage.startScan')}</Link>
            <Link to={publicPath('/#report')} className="btn btn-secondary">{t('faqPage.viewSample')}</Link>
          </div>
        </section>

        <div className="faq-layout">
          <aside className="faq-nav" aria-label={t('faqPage.navLabel')}>
            {groups.map((group) => (
              <a key={group.id} href={`#${group.id}`}>
                <span>{group.kicker}</span>
                {group.title}
              </a>
            ))}
          </aside>

          <div className="faq-groups">
            {groups.map((group, groupIndex) => (
              <section className="faq-group" id={group.id} key={group.id}>
                <div className="faq-group-head">
                  <div>
                    <span>{group.kicker}</span>
                    <h2>{group.title}</h2>
                  </div>
                  <strong>{String(groupIndex + 1).padStart(2, '0')}</strong>
                </div>
                <div className="faq-question-list">
                  {group.items.map((item) => (
                    <details className="faq-question" key={item.q}>
                      <summary>{item.q}</summary>
                      <p>{item.a}</p>
                    </details>
                  ))}
                </div>
              </section>
            ))}
          </div>
        </div>

        <section className="faq-cta">
          <h2>{t('faqPage.ctaTitle')}</h2>
          <p>{t('faqPage.ctaBody')}</p>
          <div className="hero-actions">
            <Link to="/signin?next=/scan" className="btn btn-primary">{t('faqPage.startScan')}</Link>
            <Link to={publicPath('/pricing')} className="btn btn-secondary">{t('faqPage.viewPricing')}</Link>
          </div>
        </section>

        <PageFreshness path="/faq" />
      </div>
    </main>
  );
}
