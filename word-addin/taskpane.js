// allow-hardcode: task-pane controller — UI status strings + Office.js calls,
// not scoring/matching logic. All detection happens server-side via /api/ext.
//
// NOTE: office.js is loaded unpinned (no SRI) on purpose — Microsoft serves it as
// an auto-updating evergreen script from their CDN and requires it not be pinned.

(function () {
  "use strict";

  // Same-origin in production (the pane is served from the API domain), so a
  // relative base needs no CORS. Override for a separate host if ever needed.
  var EXT = "/api/ext";
  var KEY_STORAGE = "draftproof_api_key";
  var POLL_INTERVAL_MS = 2000;
  var POLL_MAX_ATTEMPTS = 60; // ~120s ceiling

  var els = {};
  var scanning = false;
  var pendingDocName = null; // Word file name captured at scan time

  Office.onReady(function (info) {
    if (info.host !== Office.HostType.Word) return;
    cache();
    bind();
    render();
    // Live-preview the selection + word count, updating as it changes.
    Office.context.document.addHandlerAsync(
      Office.EventType.DocumentSelectionChanged,
      updateSelectionPreview
    );
    updateSelectionPreview();
    // Re-show the last scan saved in this document, if any.
    if (getKey()) restoreSavedScan();
  });

  function cache() {
    [
      "setup", "apiKey", "saveKey", "setupError",
      "scanner", "keyPrefix", "changeKey", "scanBtn", "status", "result",
      "selectionPreview", "wordCount", "docState", "headScores",
      "versionRow", "versionSelect", "newScanBtn",
    ].forEach(function (id) { els[id] = document.getElementById(id); });
  }

  function bind() {
    els.saveKey.addEventListener("click", onSaveKey);
    els.changeKey.addEventListener("click", onChangeKey);
    els.scanBtn.addEventListener("click", onScan);
    els.versionSelect.addEventListener("change", onVersionChange);
    els.newScanBtn.addEventListener("click", onNewScan);
    // Click a finding's quoted/snippet text → jump to + highlight it in the doc.
    // Delegated because the result HTML is re-rendered on every scan.
    els.result.addEventListener("click", function (e) {
      var el = e.target;
      while (el && el !== els.result && !(el.getAttribute && el.getAttribute("data-find"))) {
        el = el.parentNode;
      }
      if (!el || !el.getAttribute) return;
      var needle = el.getAttribute("data-find");
      if (needle) locateInDoc(needle, el.getAttribute("data-scope"));
    });
  }

  function getKey() {
    try { return window.localStorage.getItem(KEY_STORAGE) || ""; } catch (e) { return ""; }
  }
  function setKey(v) {
    try { v ? window.localStorage.setItem(KEY_STORAGE, v) : window.localStorage.removeItem(KEY_STORAGE); } catch (e) {}
  }

  function render() {
    var key = getKey();
    if (key) {
      els.keyPrefix.textContent = key.slice(0, 16) + "…";
      show(els.scanner); hide(els.setup);
      updateSelectionPreview();
    } else {
      show(els.setup); hide(els.scanner);
    }
  }

  function onSaveKey() {
    var v = (els.apiKey.value || "").trim();
    if (v.indexOf("dp_live_") !== 0) {
      setupError("That doesn't look like a DraftProof key (it should start with dp_live_).");
      return;
    }
    setupError(null);
    setKey(v);
    els.apiKey.value = "";
    render();
  }

  function onChangeKey() {
    setKey(null);
    clearResult();
    if (els.versionSelect) { els.versionSelect.innerHTML = ""; hide(els.versionRow); }
    render();
  }

  // Reset the pane to its clean scanning state, so it's obvious you can select
  // new text and scan again. Past versions stay saved (the picker returns after
  // the next scan / on reopen).
  function onNewScan() {
    clearResult();
    if (els.versionRow) hide(els.versionRow);
    setDocState(null);
    updateSelectionPreview();
    setStatus("Select text in your document, then tap Scan selected text.");
  }

  function onScan() {
    clearResult();
    setStatus("Reading your selection…");
    getDocName(function (docName) {
      pendingDocName = docName;
      getSelectedText()
        .then(function (text) {
          if (!text || !text.trim()) {
            setStatus("Highlight some text in the document first, then scan.");
            return null;
          }
          return submitScan(text.trim());
        })
        .catch(function (err) { setStatus(errorText(err)); });
    });
  }

  // The Word file name (basename of the document URL), or null if unsaved.
  function getDocName(cb) {
    try {
      Office.context.document.getFilePropertiesAsync(function (res) {
        var url = (res && res.value && res.value.url) || "";
        var name = "";
        if (url) {
          var base = url.split(/[\\/]/).pop().split("?")[0];
          try { name = decodeURIComponent(base); } catch (e) { name = base; }
        }
        cb(name || null);
      });
    } catch (e) { cb(null); }
  }

  // ── Office selection ───────────────────────────────────────────────────────
  function getSelectedText() {
    return new Promise(function (resolve, reject) {
      Office.context.document.getSelectedDataAsync(Office.CoercionType.Text, function (res) {
        if (res.status === Office.AsyncResultStatus.Succeeded) resolve(res.value);
        else reject(res.error);
      });
    });
  }

  function countWords(text) {
    var t = (text || "").trim();
    return t ? t.split(/\s+/).length : 0;
  }

  // Build a robust search needle from a finding's verbatim text: collapse
  // whitespace and cap length (Word's body.search rejects strings over 255).
  function findNeedle(text, maxChars) {
    var t = (text || "").replace(/\s+/g, " ").trim();
    if (t.length <= maxChars) return t;
    var cut = t.slice(0, maxChars);
    var sp = cut.lastIndexOf(" ");
    return (sp > 40 ? cut.slice(0, sp) : cut).trim();  // don't end mid-word
  }

  // Jump to + highlight the passage a finding refers to. scope "paragraph"
  // selects the whole containing paragraph (issues); else the matched run (CT).
  function locateInDoc(needle, scope) {
    if (!needle || typeof Word === "undefined" || !Word.run) return;
    Word.run(function (context) {
      var results = context.document.body.search(needle, {
        matchCase: false, ignoreSpace: true, ignorePunct: true,
      });
      results.load("items");
      return context.sync().then(function () {
        if (!results.items.length) {
          setStatus("Couldn’t find that passage in the document.");
          return context.sync();
        }
        var target = scope === "paragraph"
          ? results.items[0].paragraphs.getFirst()
          : results.items[0];
        target.select();
        setStatus("");
        return context.sync();
      });
    }).catch(function () {
      setStatus("Couldn’t highlight that passage in the document.");
    });
  }

  // Reflect the current selection (preview + word count) in the pane. Fires on
  // load and on every DocumentSelectionChanged event.
  function updateSelectionPreview() {
    getSelectedText().then(function (text) {
      var t = (text || "").trim();
      var n = countWords(t);
      els.wordCount.textContent = n === 1 ? "1 word" : n + " words";
      els.wordCount.classList.toggle("dp-wordcount-low", n > 0 && n < 100);
      if (n === 0) {
        els.selectionPreview.textContent = "Highlight text in the document to scan.";
        els.selectionPreview.classList.add("dp-muted");
      } else {
        // Show the full selection; the box is scrollable + user-resizable.
        els.selectionPreview.textContent = t;
        els.selectionPreview.classList.remove("dp-muted");
      }
      if (!scanning) els.scanBtn.disabled = n === 0;
    }).catch(function () {});
  }

  // ── API ──────────────────────────────────────────────────────────────────
  function authHeaders() {
    return { "Content-Type": "application/json", Authorization: "Bearer " + getKey() };
  }

  function submitScan(text) {
    busy(true);
    setStatus("Scanning… this can take up to a minute.");
    return fetch(EXT + "/scan", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ text: text, document_name: pendingDocName }),
    })
      .then(handleAuthThenJson)
      .then(function (data) {
        return poll(data.scan_id, data.low_confidence, 0);
      })
      .catch(function (err) {
        busy(false);
        if (err && err.handled) return; // already surfaced
        setStatus(errorText(err));
      });
  }

  function poll(scanId, lowConfidence, attempt) {
    if (attempt >= POLL_MAX_ATTEMPTS) {
      busy(false);
      setStatus("Still working — check back in the DraftProof dashboard in a moment.");
      return;
    }
    return fetch(EXT + "/scan/" + encodeURIComponent(scanId), { headers: authHeaders() })
      .then(handleAuthThenJson)
      .then(function (data) {
        if (data.status === "completed") {
          busy(false);
          fetchAndRenderReport(scanId, lowConfidence);
        } else if (data.status === "failed") {
          busy(false);
          setStatus("The scan failed. Please try again.");
        } else {
          return delay(POLL_INTERVAL_MS).then(function () {
            return poll(scanId, lowConfidence, attempt + 1);
          });
        }
      });
  }

  // Maps non-2xx to friendly messages; throws {handled:true} once surfaced.
  function handleAuthThenJson(resp) {
    if (resp.ok) return resp.json();
    if (resp.status === 401) {
      setKey(null); render();
      setupError("Your API key was rejected. Paste a current key from draftproof.app/api-keys.");
      throw { handled: true };
    }
    if (resp.status === 402) {
      busy(false);
      setStatus("You're out of credits. Top up at draftproof.app/buy, then scan again.");
      throw { handled: true };
    }
    return resp.json().catch(function () { return {}; }).then(function (body) {
      throw new Error((body && body.detail) || ("Request failed (" + resp.status + ")"));
    });
  }

  // ── Rendering ──────────────────────────────────────────────────────────────
  // After a scan completes, pull the richer report (Critical Thinking, Submitted
  // content, signal highlights) and render + persist it into the document.
  function fetchAndRenderReport(scanId, lowConfidence) {
    setStatus("Loading results…");
    fetch(EXT + "/scan/" + encodeURIComponent(scanId) + "/report", { headers: authHeaders() })
      .then(handleAuthThenJson)
      .then(function (report) {
        // Record this scan as a new version FIRST so we can label the result with it.
        var entry = recordScan(scanId, report, lowConfidence, pendingDocName);
        renderReport(report, {
          lowConfidence: lowConfidence, restored: false, docName: pendingDocName,
          version: entry.version, savedAt: formatTs(entry.ts),
        });
      })
      .catch(function (err) {
        if (err && err.handled) return;
        setStatus(errorText(err));
      });
  }

  function pct(v) { return v == null ? "—" : Math.round(v) + "%"; }

  // allow-hardcode: HTML render template + class names for the header score
  // badges (presentation), not scoring logic.
  function setHeadScores(report) {
    if (!els.headScores) return;
    var ai = report && report.ai_score != null ? pct(report.ai_score) : null;
    var wq = report && report.writing_score != null ? pct(report.writing_score) : null;
    if (ai == null && wq == null) {
      els.headScores.hidden = true;
      els.headScores.innerHTML = "";
      return;
    }
    var html = "";
    if (ai != null) html += scoreBadge("AI", "AI-likelihood", ai);
    if (wq != null) html += scoreBadge("Writing", "Writing quality", wq);
    els.headScores.innerHTML = html;
    els.headScores.hidden = false;
  }

  function scoreBadge(label, full, value) {
    return '<span class="dp-score-badge" title="' + escapeHtml(full) + '">' +
      '<span class="dp-score-label">' + escapeHtml(label) + '</span>' +
      '<span class="dp-score-value">' + escapeHtml(value) + "</span></span>";
  }

  function section(title, valueHtml, levelClass, noteText) {
    return '<div class="dp-section"><div class="dp-section-title">' + escapeHtml(title) + "</div>" +
      '<div class="dp-section-value' + (levelClass ? " " + levelClass : "") + '">' + valueHtml + "</div>" +
      (noteText ? '<p class="dp-section-note">' + escapeHtml(noteText) + "</p>" : "") + "</div>";
  }

  function renderReport(report, opts) {
    opts = opts || {};
    setStatus("");
    var p = [];

    setHeadScores(report);  // AI-likelihood + writing-quality badges live in the header row

    // Show the text this result is about in the "Selected text" box. For a
    // restored result there is no live selection; for a fresh one this keeps the
    // result tied to exactly what was scanned. A later selection change overwrites it.
    if (report.scanned_text) {
      els.selectionPreview.textContent = report.scanned_text;
      els.selectionPreview.classList.remove("dp-muted");
      var swc = report.word_count != null ? report.word_count : countWords(report.scanned_text);
      els.wordCount.textContent = swc === 1 ? "1 word" : swc + " words";
      els.wordCount.classList.toggle("dp-wordcount-low", swc > 0 && swc < 100);
    }

    // Version caption — which scan you're viewing (set for both fresh + restored).
    if (opts.version) {
      p.push('<p class="dp-version-caption">Version ' + opts.version + ".0" +
        (opts.savedAt ? ' · scanned ' + escapeHtml(opts.savedAt) : "") +
        (opts.restored ? "" : " · new") + "</p>");
    }

    // allow-hardcode: the strings below are HTML render templates + CSS class names
    // for presentation; all user-facing text comes from the server report. No
    // scoring/matching list here.
    var ct = report.critical_thinking;
    if (ct && ct.questions && ct.questions.length) {
      var qHtml = ct.questions.map(function (q, i) {
        var quote = q.quote
          ? '<p class="dp-q-quote dp-locate" data-scope="text" data-find="' +
            escapeAttr(findNeedle(q.quote, 200)) +
            '" title="Click to highlight this in your document">“' +
            escapeHtml(q.quote) + '”</p>'
          : "";
        return '<li><span class="dp-q-num">' + (i + 1) + "</span>" +
          "<div>" + quote + '<p class="dp-q-text">' + escapeHtml(q.question) + "</p></div></li>";
      }).join("");
      p.push('<div class="dp-section"><div class="dp-section-title">Critical thinking</div>' +
        '<p class="dp-section-note">Questions to sharpen your thinking — the answers are yours to write.</p>' +
        '<ul class="dp-q-list">' + qHtml + "</ul></div>");
    } else if (ct && (ct.status || ct.score != null)) {
      var ctHead = escapeHtml(ct.status || "—") +
        (ct.score != null ? " · " + Math.round(ct.score) + "/100" : "");
      var ctBody = "";
      if (ct.action) {
        ctBody += '<p class="dp-section-note"><strong>' + escapeHtml(ct.lead || "Focus") +
          ":</strong> " + escapeHtml(ct.action) + "</p>";
      }
      if (ct.dimensions && ct.dimensions.length) {
        ctBody += '<ul class="dp-dim-list">' + ct.dimensions.map(function (d) {
          return "<li><span>" + escapeHtml(d.label || "") + "</span>" +
            '<span class="dp-dim-score">' + Math.round(d.control) + "/100</span></li>";
        }).join("") + "</ul>";
      }
      if (ct.caveat) {
        ctBody += '<p class="dp-section-note dp-muted">' + escapeHtml(ct.caveat) + "</p>";
      }
      p.push('<div class="dp-section"><div class="dp-section-title">Critical thinking</div>' +
        '<div class="dp-section-value">' + ctHead + "</div>" + ctBody + "</div>");
    }

    var sr = report.submission_risk;
    if (sr && (sr.label || sr.level)) {
      var srHead = escapeHtml(sr.label || sr.level) + (sr.risk != null ? " · " + Math.round(sr.risk) + "%" : "");
      p.push(section("Submitted content", srHead, "dp-level-" + escapeAttr(sr.level || "unknown"), sr.reason));
    }

    // allow-hardcode: HTML render templates + CSS class names (presentation); all
    // user-facing text comes from the server report, no scoring/matching list.
    var pis = report.paragraph_issues || [];
    if (pis.length) {
      var cards = pis.map(function (it) {
        var chips = "";
        if (it.tier) {
          chips += '<span class="dp-tier dp-tier-' + escapeAttr(it.tier) + ' dp-issue-tier">' +
            escapeHtml(it.tier) + "</span>";
        }
        if (it.signal_label) chips += '<span class="dp-issue-sig">' + escapeHtml(humanize(it.signal_label)) + "</span>";
        var b = "";
        if (it.snippet) {
          b += '<p class="dp-issue-snippet dp-locate" data-scope="paragraph" data-find="' +
            escapeAttr(findNeedle(it.snippet, 120)) +
            '" title="Click to highlight this paragraph in your document">' +
            escapeHtml(it.snippet) + "</p>";
        }
        if (it.reader_summary) b += '<p class="dp-issue-summary">' + escapeHtml(it.reader_summary) + "</p>";
        if (it.main_issue) {
          b += '<div class="dp-issue-block"><span class="dp-issue-label">Main issue to fix</span><p>' +
            escapeHtml(it.main_issue) + "</p></div>";
        }
        if (it.recommendation) {
          b += '<div class="dp-issue-block"><span class="dp-issue-label">How to improve</span><p>' +
            escapeHtml(it.recommendation) + "</p></div>";
        }
        return '<article class="dp-issue-card">' +
          (chips ? '<div class="dp-issue-chips">' + chips + "</div>" : "") + b + "</article>";
      }).join("");
      p.push('<div class="dp-section"><div class="dp-section-title">Issues</div>' + cards + "</div>");
    } else {
      var sig = report.signal_highlights || [];
      if (sig.length) {
        var items = sig.map(function (s) {
          var title = humanize(s.title || s.description || "");
          var desc = (s.description && s.description !== s.title)
            ? '<p class="dp-sig-desc">' + escapeHtml(s.description) + "</p>" : "";
          var rec = s.recommendation
            ? '<p class="dp-sig-rec">' + escapeHtml(s.recommendation) + "</p>" : "";
          return '<li><span class="dp-sev dp-sev-' + escapeAttr(s.severity || "low") + '"></span>' +
            '<div class="dp-sig-body"><span class="dp-sig-title">' + escapeHtml(title) + "</span>" +
            desc + rec + "</div></li>";
        }).join("");
        p.push('<div class="dp-section"><div class="dp-section-title">Signal highlights</div>' +
          '<ul class="dp-signal-list">' + items + "</ul></div>");
      }
    }

    if (opts.lowConfidence) {
      p.push('<p class="dp-lowconf">Short selection — this read is indicative only. Scan a fuller passage (100+ words) for a confident result.</p>');
    }

    if (report.report_url) {
      p.push('<a class="dp-report-link" href="https://draftproof.app' +
        escapeHtml(report.report_url) + '" target="_blank" rel="noopener">View full report ↗</a>');
    }

    els.result.innerHTML = p.join("");
    show(els.result);
  }

  // "medium_predictability" -> "Medium predictability"
  function humanize(s) {
    return String(s || "").replace(/_/g, " ").replace(/^./, function (c) { return c.toUpperCase(); });
  }

  // ── Persist the last scan into the Word document ────────────────────────────
  // Office document settings travel with the .docx, so reopening the doc lets us
  // re-show the report without rescanning (offline snapshot + link to the web).
  var DOC_SETTING = "draftproof_scan_v1";        // legacy single-snapshot (migrated)
  var HISTORY_SETTING = "draftproof_history_v1";  // { scans: [meta...], latestReport }
  var MAX_HISTORY = 25;                           // cap stored versions (metadata is tiny)

  // Read the per-document scan history, migrating the old single-snapshot key once.
  function loadHistory() {
    var h = null;
    try { h = Office.context.document.settings.get(HISTORY_SETTING); } catch (e) {}
    if (h && h.scans) return h;
    var old = null;
    try { old = Office.context.document.settings.get(DOC_SETTING); } catch (e) {}
    if (old && old.scanId) {
      return {
        scans: [{ scanId: old.scanId, ts: old.ts, version: 1, lowConfidence: !!old.lowConfidence, docName: old.docName || null }],
        latestReport: old.report || null,
      };
    }
    return { scans: [], latestReport: null };
  }

  function saveHistory(hist, onOk) {
    try {
      Office.context.document.settings.set(HISTORY_SETTING, hist);
      Office.context.document.settings.saveAsync(function (res) {
        // saveAsync writes the bag INTO the document; it reaches disk (survives a
        // reopen) only when the FILE itself is saved — so persist the file too.
        if (res && res.status === Office.AsyncResultStatus.Succeeded) {
          if (onOk) onOk();
          persistDocumentFile();
        } else {
          var msg = res && res.error ? res.error.message : "unknown error";
          setDocState("Couldn't save to this document: " + msg, "err");
        }
      });
    } catch (e) {
      setDocState("Couldn't save to this document: " + (e && e.message), "err");
    }
  }

  // Append a completed scan as the next version (newest), keep the report inline
  // as an offline fallback, cap the list, persist, and refresh the picker.
  // Returns the stored entry { scanId, ts, version, ... }.
  function recordScan(scanId, report, lowConfidence, docName) {
    var hist = loadHistory();
    var entry = null;
    for (var i = 0; i < hist.scans.length; i++) {
      if (hist.scans[i].scanId === scanId) entry = hist.scans[i];
    }
    if (!entry) {
      var version = 1;
      for (var j = 0; j < hist.scans.length; j++) {
        if (hist.scans[j].version >= version) version = hist.scans[j].version + 1;
      }
      entry = { scanId: scanId, ts: new Date().toISOString(), version: version, lowConfidence: !!lowConfidence, docName: docName || null };
      hist.scans.push(entry);
      if (hist.scans.length > MAX_HISTORY) {
        hist.scans.sort(function (a, b) { return a.version - b.version; });
        hist.scans = hist.scans.slice(hist.scans.length - MAX_HISTORY);
      }
    }
    hist.latestReport = report;  // only the newest report is cached inline
    saveHistory(hist, function () { setDocState("Saved v" + entry.version + ".0 to this document.", "ok"); });
    populateVersionSelect(hist.scans, scanId);
    return entry;
  }

  // Fill the version dropdown, newest first. Label is version-only; the scan time
  // rides along as the option tooltip.
  function populateVersionSelect(scans, selectedScanId) {
    if (!els.versionSelect) return;
    if (!scans || !scans.length) { hide(els.versionRow); els.versionSelect.innerHTML = ""; return; }
    var sorted = scans.slice().sort(function (a, b) { return b.version - a.version; });
    var html = "";
    for (var i = 0; i < sorted.length; i++) {
      var s = sorted[i];
      html += '<option value="' + escapeAttr(s.scanId) + '"' +
        (s.scanId === selectedScanId ? " selected" : "") +
        ' title="' + escapeAttr(formatTs(s.ts)) + '">v' + s.version + '.0</option>';
    }
    els.versionSelect.innerHTML = html;
    show(els.versionRow);
  }

  function onVersionChange() {
    var scanId = els.versionSelect.value;
    if (!scanId) return;
    var hist = loadHistory();
    var meta = null, newest = null;
    for (var i = 0; i < hist.scans.length; i++) {
      var s = hist.scans[i];
      if (s.scanId === scanId) meta = s;
      if (!newest || s.version > newest.version) newest = s;
    }
    var isNewest = newest && newest.scanId === scanId;
    var vlabel = meta ? ("v" + meta.version + ".0") : "this version";
    setStatus("Loading " + vlabel + "…");
    var opts = {
      lowConfidence: meta && meta.lowConfidence, restored: true,
      savedAt: meta ? formatTs(meta.ts) : "", docName: meta ? meta.docName : null,
      version: meta ? meta.version : null,
    };
    // Re-fetch the live report by scan id; fall back to the inline snapshot only
    // for the newest scan (the only one cached) if the network/report is gone.
    fetch(EXT + "/scan/" + encodeURIComponent(scanId) + "/report", { headers: authHeaders() })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
      .then(function (fresh) { setStatus(""); renderReport(fresh, opts); })
      .catch(function () {
        if (isNewest && hist.latestReport) { setStatus(""); renderReport(hist.latestReport, opts); }
        else { setStatus("Couldn't load " + vlabel + " — it may be offline."); }
      });
  }

  function setDocState(msg, kind) {
    if (!els.docState) return;
    if (!msg) { els.docState.hidden = true; els.docState.textContent = ""; return; }
    els.docState.textContent = msg;
    els.docState.className = "dp-docstate" + (kind ? " dp-docstate-" + kind : "");
    els.docState.hidden = false;
  }

  // Save the document file so the in-document settings reach disk — but only if
  // it already has a path, to avoid springing a "Save As" dialog on a brand-new
  // unsaved doc (where in-session restore still works until the user saves once).
  function persistDocumentFile() {
    try {
      Office.context.document.getFilePropertiesAsync(function (res) {
        var url = res && res.value && res.value.url;
        if (!url || typeof Word === "undefined" || !Word.run) return;
        Word.run(function (context) {
          context.document.save();
          return context.sync();
        }).catch(function () {});
      });
    } catch (e) { /* non-fatal */ }
  }

  // On open: populate the version picker from this document's history and show
  // the newest scan. Switching versions is handled by onVersionChange.
  function restoreSavedScan() {
    var hist = loadHistory();
    if (!hist.scans.length) {
      setDocState("No saved scan found in this document yet.", null);
      populateVersionSelect([], null);
      return;
    }
    var newest = hist.scans.slice().sort(function (a, b) { return b.version - a.version; })[0];
    populateVersionSelect(hist.scans, newest.scanId);
    onVersionChange();  // loads the (now-selected) newest version
  }

  function formatTs(iso) {
    try { return new Date(iso).toLocaleString(); } catch (e) { return ""; }
  }

  // ── tiny helpers ───────────────────────────────────────────────────────────
  function show(el) { if (el) el.hidden = false; }
  function hide(el) { if (el) el.hidden = true; }
  function busy(on) { scanning = on; els.scanBtn.disabled = on; }
  function setStatus(msg) { els.status.textContent = msg || ""; }
  function clearResult() { els.result.innerHTML = ""; hide(els.result); setStatus(""); setHeadScores(null); }
  function setupError(msg) {
    if (!msg) { els.setupError.hidden = true; els.setupError.textContent = ""; return; }
    els.setupError.textContent = msg; els.setupError.hidden = false;
  }
  function delay(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function errorText(err) {
    return (err && (err.message || err.name)) ? "Couldn't scan: " + (err.message || err.name) : "Something went wrong. Please try again.";
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function escapeAttr(s) { return String(s).replace(/[^a-z0-9_-]/gi, ""); }
})();
