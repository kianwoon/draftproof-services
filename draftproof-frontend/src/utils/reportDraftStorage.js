const DB_NAME = 'draftproof-report-drafts';
const DB_VERSION = 1;
const STORE_NAME = 'drafts';
const LOCAL_PREFIX = 'draftproof_report_draft:';

function canUseIndexedDb() {
  return typeof window !== 'undefined' && 'indexedDB' in window;
}

function openDraftDb() {
  if (!canUseIndexedDb()) return Promise.reject(new Error('IndexedDB unavailable'));

  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'reportId' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Failed to open draft storage'));
  });
}

function withStore(mode, operation) {
  return openDraftDb().then((db) => new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, mode);
    const store = transaction.objectStore(STORE_NAME);
    const request = operation(store);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Draft storage operation failed'));
    transaction.oncomplete = () => db.close();
    transaction.onerror = () => {
      db.close();
      reject(transaction.error || new Error('Draft storage transaction failed'));
    };
  }));
}

function localKey(reportId) {
  return `${LOCAL_PREFIX}${reportId}`;
}

function readLocalDraft(reportId) {
  try {
    const raw = window.localStorage.getItem(localKey(reportId));
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function writeLocalDraft(reportId, draft) {
  try {
    window.localStorage.setItem(localKey(reportId), JSON.stringify(draft));
  } catch {
    // Browser storage can be unavailable in private mode or under quota pressure.
  }
}

function removeLocalDraft(reportId) {
  try {
    window.localStorage.removeItem(localKey(reportId));
  } catch {
    // Ignore local cleanup failures.
  }
}

export async function getReportDraft(reportId) {
  if (!reportId) return null;
  try {
    return await withStore('readonly', (store) => store.get(String(reportId)));
  } catch {
    return readLocalDraft(reportId);
  }
}

export async function saveReportDraft(reportId, text) {
  if (!reportId) return null;
  const draft = {
    reportId: String(reportId),
    text,
    updatedAt: new Date().toISOString(),
  };

  try {
    await withStore('readwrite', (store) => store.put(draft));
  } catch {
    writeLocalDraft(reportId, draft);
  }

  return draft;
}

export async function deleteReportDraft(reportId) {
  if (!reportId) return;
  try {
    await withStore('readwrite', (store) => store.delete(String(reportId)));
  } catch {
    removeLocalDraft(reportId);
  }
  removeLocalDraft(reportId);
}
