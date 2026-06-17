import { useTranslation } from 'react-i18next';
import { announcements } from '../announcements';

// Static featured-announcement band pinned under the header (replaces the old
// scrolling NewsTicker). Renders the FIRST announcement as a rich band: a
// source/date chip, a bold headline + supporting line, an outbound "Read
// update" link, and a row of topic pills. Driven by announcements.js so the
// copy stays editable in one place. The band height is reserved globally via
// --ticker-h in .app-main, so .announce-band is pinned to that same height.
export default function AnnouncementBanner() {
  const { t } = useTranslation();

  if (!Array.isArray(announcements) || announcements.length === 0) return null;

  const item = announcements[0];
  if (!item) return null;

  const pills = Array.isArray(item.pills) ? item.pills : [];
  const chip = [item.badge, item.date].filter(Boolean).join(' · ');

  return (
    <aside className="announce-band" aria-label={t('ticker.label')}>
      <div className="announce-band-inner">
        <div className="announce-band-lead">
          {chip && <span className="announce-band-chip">{chip}</span>}
          <p className="announce-band-text">
            {item.headline && <strong>{item.headline} </strong>}
            <AnnounceText text={item.text} emphasis={item.emphasis} />
          </p>
          {item.url && (
            <a
              className="announce-band-cta"
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
            >
              {t('ticker.readUpdate')}
            </a>
          )}
        </div>

        {pills.length > 0 && (
          <ul className="announce-band-pills">
            {pills.map((pill) => (
              <li key={pill}>{pill}</li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}

// Renders the supporting line, italicising a single emphasised word if one is
// supplied and present in the text (purely typographic — no logic depends on it).
function AnnounceText({ text, emphasis }) {
  if (!text) return null;
  if (!emphasis || !text.includes(emphasis)) return text;

  const index = text.indexOf(emphasis);
  return (
    <>
      {text.slice(0, index)}
      <em>{emphasis}</em>
      {text.slice(index + emphasis.length)}
    </>
  );
}
