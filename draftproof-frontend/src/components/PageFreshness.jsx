import { useTranslation } from 'react-i18next';
import {
  formatFreshnessDate,
  getFreshnessLabelKey,
  getSeoMeta,
} from '../seoMetadata';

function useFreshness(path) {
  const { t, i18n } = useTranslation();
  const meta = getSeoMeta(path, t);
  if (!meta.freshness?.date) return null;

  return {
    date: meta.freshness.date,
    displayDate: formatFreshnessDate(meta.freshness.date, i18n.resolvedLanguage),
    label: t(getFreshnessLabelKey(meta)),
  };
}

export default function PageFreshness({ path, className = '' }) {
  const freshness = useFreshness(path);
  if (!freshness) return null;

  const classes = ['page-freshness', className].filter(Boolean).join(' ');
  return (
    <p className={classes}>
      <span>{freshness.label}</span>
      <time dateTime={freshness.date}>{freshness.displayDate}</time>
    </p>
  );
}

export function FreshnessStat({ path, detail }) {
  const freshness = useFreshness(path);
  if (!freshness) return null;

  return (
    <div className="app-hero-stat">
      <span>{freshness.label}</span>
      <strong><time dateTime={freshness.date}>{freshness.displayDate}</time></strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}
