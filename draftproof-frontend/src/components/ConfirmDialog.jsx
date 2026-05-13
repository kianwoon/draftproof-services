import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export default function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = null,
  cancelLabel = null,
  confirmClassName = 'btn-danger',
  hideCancel = false,
  onConfirm,
  onCancel,
}) {
  const dialogRef = useRef(null);
  const { t } = useTranslation();

  useEffect(() => {
    if (open) dialogRef.current?.focus();
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handleKey = (e) => {
      if (e.key === 'Escape') onCancel();
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <h3 className="modal-title">{title}</h3>
        <p className="modal-message">{message}</p>
        <div className="modal-actions">
          {!hideCancel && (
            <button className="btn btn-secondary btn-small" onClick={onCancel}>{cancelLabel || t('dialog.cancel')}</button>
          )}
          <button className={`btn btn-small ${confirmClassName}`} ref={dialogRef} onClick={onConfirm}>{confirmLabel || t('dialog.confirm')}</button>
        </div>
      </div>
    </div>
  );
}
