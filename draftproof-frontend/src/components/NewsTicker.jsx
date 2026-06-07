import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { announcements } from '../announcements';

// Continuously scrolling site-news bar pinned under the header.
//
// The scroll is driven by requestAnimationFrame (not a CSS keyframe) so it
// keeps running on mobile: CSS marquees get frozen by prefers-reduced-motion
// and are unreliable on iOS Safari when combined with mask-image + a
// content-sized track. The track renders the item list twice and we wrap the
// pixel offset at half the track width for a seamless loop. Scrolling pauses
// while a pointer is over the bar and while the tab is hidden.
export default function NewsTicker() {
  const { t } = useTranslation();
  const trackRef = useRef(null);
  const pausedRef = useRef(false);

  useEffect(() => {
    const track = trackRef.current;
    if (!track) return undefined;

    const SPEED = 45; // px per second
    let raf = 0;
    let last = 0;
    let offset = 0;

    const step = (ts) => {
      const delta = last ? (ts - last) / 1000 : 0;
      last = ts;
      if (!pausedRef.current && document.visibilityState === 'visible') {
        const half = track.scrollWidth / 2;
        if (half > 0) {
          offset -= SPEED * delta;
          if (-offset >= half) offset += half; // seamless wrap
          track.style.transform = `translate3d(${offset}px, 0, 0)`;
        }
      }
      raf = window.requestAnimationFrame(step);
    };

    raf = window.requestAnimationFrame(step);
    return () => window.cancelAnimationFrame(raf);
  }, []);

  if (!Array.isArray(announcements) || announcements.length === 0) return null;

  const renderItem = (item, key) => {
    const body = (
      <>
        <span className="news-ticker-dot" aria-hidden="true" />
        {item.date && <span className="news-ticker-date">{item.date}</span>}
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

  const pause = () => { pausedRef.current = true; };
  const resume = () => { pausedRef.current = false; };

  return (
    <aside
      className="news-ticker"
      aria-label={t('ticker.label')}
      onPointerEnter={pause}
      onPointerLeave={resume}
    >
      <span className="news-ticker-badge">{t('ticker.badge')}</span>
      <div className="news-ticker-viewport">
        <div className="news-ticker-track" ref={trackRef}>
          {announcements.map((item, i) => renderItem(item, `a-${item.id}-${i}`))}
          {announcements.map((item, i) => renderItem(item, `b-${item.id}-${i}`))}
        </div>
      </div>
    </aside>
  );
}
