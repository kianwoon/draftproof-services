/**
 * DraftProof — Google Docs editor add-on (server side).
 *
 * Calls the same key-authenticated endpoints as the Word add-in
 * (https://draftproof.app/api/ext/*) but SERVER-SIDE via UrlFetchApp, so there's
 * no CORS and the API key lives in PropertiesService (never in client JS).
 * The sidebar (Sidebar.html) talks to these functions via google.script.run.
 */

var API_BASE = 'https://draftproof.app/api/ext';
var KEY_PROP = 'draftproof_api_key';
var KEY_PREFIX = 'dp_live_';

// ── Add-on lifecycle ─────────────────────────────────────────────────────────
function onOpen() {
  DocumentApp.getUi()
    .createAddonMenu()
    .addItem('Open DraftProof', 'showSidebar')
    .addToUi();
}

function onInstall() {
  onOpen();
}

function showSidebar() {
  var html = HtmlService.createHtmlOutputFromFile('Sidebar').setTitle('DraftProof');
  DocumentApp.getUi().showSidebar(html);
}

// ── API key (per-user, server-side) ─────────────────────────────────────────
function getKeyInfo() {
  var key = PropertiesService.getUserProperties().getProperty(KEY_PROP);
  if (!key) return { hasKey: false };
  return { hasKey: true, prefix: key.substring(0, 16) };
}

function saveKey(key) {
  key = (key || '').trim();
  if (key.indexOf(KEY_PREFIX) !== 0) {
    throw new Error("That doesn't look like a DraftProof key (it should start with dp_live_).");
  }
  PropertiesService.getUserProperties().setProperty(KEY_PROP, key);
  return getKeyInfo();
}

function clearKey() {
  PropertiesService.getUserProperties().deleteProperty(KEY_PROP);
  return { hasKey: false };
}

// ── Selection ────────────────────────────────────────────────────────────────
function getSelectionText() {
  var doc = DocumentApp.getActiveDocument();
  var selection = doc.getSelection();
  if (!selection) return '';
  var parts = [];
  var elements = selection.getRangeElements();
  for (var i = 0; i < elements.length; i++) {
    var re = elements[i];
    var el = re.getElement();
    if (re.isPartial()) {
      var txt = el.asText().getText();
      parts.push(txt.substring(re.getStartOffset(), re.getEndOffsetInclusive() + 1));
    } else if (el.editAsText) {
      parts.push(el.asText().getText());
    } else if (el.getText) {
      parts.push(el.getText());
    }
  }
  return parts.join('\n').trim();
}

function previewSelection() {
  var text = getSelectionText();
  var wc = text ? text.split(/\s+/).filter(Boolean).length : 0;
  return { text: text.length > 400 ? text.substring(0, 400) : text, wordCount: wc };
}

// ── Scan (submit → poll → report), mirroring the Word flow ───────────────────
function submitScan() {
  var text = getSelectionText();
  if (!text) throw new Error('NO_SELECTION');
  var wc = text.split(/\s+/).filter(Boolean).length;
  var data = _request('/scan', 'post', { text: text, document_name: _docName() });
  return { scanId: data.scan_id, wordCount: wc, lowConfidence: data.low_confidence };
}

function getScanStatus(scanId) {
  return _request('/scan/' + encodeURIComponent(scanId), 'get');
}

function getScanReport(scanId) {
  return _request('/scan/' + encodeURIComponent(scanId) + '/report', 'get');
}

function getCredits() {
  return _request('/credits', 'get');
}

// ── Per-document scan history (version picker) ───────────────────────────────
// Stored in document properties so it travels with the doc and survives reopen.
var SCANS_PROP = 'draftproof_scans';
var MAX_HISTORY = 25;

function _readScans() {
  try {
    var raw = PropertiesService.getDocumentProperties().getProperty(SCANS_PROP);
    return raw ? JSON.parse(raw) : [];
  } catch (e) { return []; }
}

// Newest first.
function listScans() {
  var arr = _readScans();
  arr.sort(function (a, b) { return b.version - a.version; });
  return arr;
}

// Append a completed scan as the next version, cap, persist. Returns the full
// list (newest first) so the sidebar can repopulate the picker.
function recordScan(scanId, lowConfidence) {
  var props = PropertiesService.getDocumentProperties();
  var arr = _readScans();
  var exists = false;
  for (var i = 0; i < arr.length; i++) { if (arr[i].scanId === scanId) exists = true; }
  if (!exists) {
    var version = 1;
    for (var j = 0; j < arr.length; j++) { if (arr[j].version >= version) version = arr[j].version + 1; }
    arr.push({ scanId: scanId, ts: Date.now(), version: version, lowConfidence: !!lowConfidence });
    arr.sort(function (a, b) { return a.version - b.version; });
    if (arr.length > MAX_HISTORY) arr = arr.slice(arr.length - MAX_HISTORY);
    props.setProperty(SCANS_PROP, JSON.stringify(arr));
  }
  arr.sort(function (a, b) { return b.version - a.version; });
  return arr;
}

// Jump to + highlight the passage a finding refers to. scope 'paragraph'
// selects the whole containing paragraph (issues); else the matched run (CT).
// Returns true if located. findText takes a regex subset, so metacharacters are
// escaped and straight/curly quotes made interchangeable (verbatim doc text).
function highlightInDoc(needle, scope) {
  if (!needle) return false;
  try {
    var doc = DocumentApp.getActiveDocument();
    var body = doc.getBody();
    var rx = String(needle)
      .replace(/[\\^$.*+?()[\]{}|]/g, '\\$&')
      .replace(/['‘’]/g, "['‘’]")
      .replace(/["“”]/g, '["“”]');
    var found = body.findText(rx);
    if (!found) return false;
    var textEl = found.getElement();
    var range = doc.newRange();
    if (scope === 'paragraph') {
      range.addElement(textEl.getParent());          // whole paragraph / list item
    } else {
      range.addElement(textEl.asText(), found.getStartOffset(), found.getEndOffsetInclusive());
    }
    doc.setSelection(range.build());
    return true;
  } catch (e) {
    return false;
  }
}

// ── helpers ──────────────────────────────────────────────────────────────────
function _docName() {
  try { return DocumentApp.getActiveDocument().getName(); } catch (e) { return null; }
}

function _key() {
  var k = PropertiesService.getUserProperties().getProperty(KEY_PROP);
  if (!k) throw new Error('NO_KEY');
  return k;
}

function _request(path, method, payload) {
  var options = {
    method: method || 'get',
    muteHttpExceptions: true,
    contentType: 'application/json',
    headers: { Authorization: 'Bearer ' + _key() },
  };
  if (payload) options.payload = JSON.stringify(payload);

  var res = UrlFetchApp.fetch(API_BASE + path, options);
  var code = res.getResponseCode();
  var body = res.getContentText();
  var data = {};
  try { data = JSON.parse(body); } catch (e) { /* non-JSON error body */ }

  if (code === 401) throw new Error('BAD_KEY');
  if (code === 402) throw new Error('NO_CREDITS');
  if (code >= 400) throw new Error((data && data.detail) || ('Request failed (' + code + ')'));
  return data;
}
