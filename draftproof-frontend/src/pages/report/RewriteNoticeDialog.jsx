import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export default function RewriteNoticeDialog({ open, title, message, onClose }) {
  const { t } = useTranslation();
  const closeButtonRef = useRef(null);

  useEffect(() => {
    if (open) {
      closeButtonRef.current?.focus();
    }
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="rewrite-notice-title"
        onClick={(event) => event.stopPropagation()}
      >
        <h3 id="rewrite-notice-title" className="modal-title">{title}</h3>
        <p className="modal-message">{message}</p>
        <div className="modal-actions">
          <button
            ref={closeButtonRef}
            type="button"
            className="btn btn-primary btn-small"
            onClick={onClose}
          >
            {t('report.ok')}
          </button>
        </div>
      </div>
    </div>
  );
}
