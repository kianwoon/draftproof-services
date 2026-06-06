import { useTranslation } from 'react-i18next';
import { announcements } from '../announcements';

// Continuously scrolling site-news bar pinned under the header. The track
// renders the item sequence twice so the CSS marquee can loop seamlessly
// (translateX(-50%) lands copy B exactly where copy A began). Scrolling
// pauses on hover/focus and is disabled under prefers-reduced-motion.
export default function NewsTicker() {
  const { t } = useTranslation();

  if (!Array.isArray(announcements) || announcements.length === 0) return null;

  const renderItem = (item, key) => {
    const body = (
      <>
        <span className="news-ticker-dot" aria-hidden="true" />
        <span className="news-ticker-text">{item.text}</span>
        {item.url && <span className="news-ticker-cta">{t('ticker.readMore')}</span>}
      </>
    );

    return item.url ? (
      <a
        key={key}
        className="news-ticker-item is-link"
        href={item.url}
        target="_blank"
        rel="noopener noreferrer"
      >
        {body}
      </a>
    ) : (
      <span key={key} className="news-ticker-item">{body}</span>
    );
  };

  return (
    <aside className="news-ticker" aria-label={t('ticker.label')}>
      <span className="news-ticker-badge">{t('ticker.badge')}</span>
      <div className="news-ticker-viewport">
        <div className="news-ticker-track">
          {announcements.map((item, i) => renderItem(item, `a-${item.id}-${i}`))}
          {announcements.map((item, i) => renderItem(item, `b-${item.id}-${i}`))}
        </div>
      </div>
    </aside>
  );
}
