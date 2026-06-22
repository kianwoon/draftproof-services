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
      "selectionPreview", "wordCount", "docState",
    ].forEach(function (id) { els[id] = document.getElementById(id); });
  }

  function bind() {
    els.saveKey.addEventListener("click", onSaveKey);
    els.changeKey.addEventListener("click", onChangeKey);
    els.scanBtn.addEventListener("click", onScan);
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
    render();
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
        renderReport(report, { lowConfidence: lowConfidence, restored: false, docName: pendingDocName });
        saveScanToDoc(scanId, report, lowConfidence, pendingDocName);
      })
      .catch(function (err) {
        if (err && err.handled) return;
        setStatus(errorText(err));
      });
  }

  function pct(v) { return v == null ? "—" : Math.round(v) + "%"; }

  function section(title, valueHtml, levelClass, noteText) {
    return '<div class="dp-section"><div class="dp-section-title">' + escapeHtml(title) + "</div>" +
      '<div class="dp-section-value' + (levelClass ? " " + levelClass : "") + '">' + valueHtml + "</div>" +
      (noteText ? '<p class="dp-section-note">' + escapeHtml(noteText) + "</p>" : "") + "</div>";
  }

  function renderReport(report, opts) {
    opts = opts || {};
    setStatus("");
    var p = [];

    if (opts.docName) {
      p.push('<div class="dp-docname" title="' + escapeHtml(opts.docName) + '">' +
        escapeHtml(opts.docName) + "</div>");
    }

    if (opts.restored) {
      p.push('<div class="dp-restored">Last scan for this document' +
        (opts.savedAt ? " · " + escapeHtml(opts.savedAt) : "") + "</div>");
    }

    var tier = report.tier || "unknown";
    p.push('<div class="dp-tier dp-tier-' + escapeAttr(tier) + '">' + escapeHtml(tierLabel(tier)) + "</div>");
    p.push('<dl class="dp-metrics">' +
      "<dt>AI-likelihood</dt><dd>" + pct(report.ai_score) + "</dd>" +
      "<dt>Writing quality</dt><dd>" + pct(report.writing_score) + "</dd></dl>");

    // allow-hardcode: the strings below are HTML render templates + CSS class names
    // for presentation; all user-facing text comes from the server report. No
    // scoring/matching list here.
    var ct = report.critical_thinking;
    if (ct && (ct.status || ct.score != null)) {
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

  function tierLabel(tier) {
    switch (tier) {
      case "clean": return "Clean";
      case "acceptable": return "Acceptable";
      case "concerning": return "Concerning";
      case "strong": return "Strong AI signal";
      default: return tier ? tier.charAt(0).toUpperCase() + tier.slice(1) : "Unknown";
    }
  }

  // "medium_predictability" -> "Medium predictability"
  function humanize(s) {
    return String(s || "").replace(/_/g, " ").replace(/^./, function (c) { return c.toUpperCase(); });
  }

  // ── Persist the last scan into the Word document ────────────────────────────
  // Office document settings travel with the .docx, so reopening the doc lets us
  // re-show the report without rescanning (offline snapshot + link to the web).
  var DOC_SETTING = "draftproof_scan_v1";

  function saveScanToDoc(scanId, report, lowConfidence, docName) {
    try {
      Office.context.document.settings.set(DOC_SETTING, {
        scanId: scanId,
        ts: new Date().toISOString(),
        lowConfidence: !!lowConfidence,
        docName: docName || null,
        report: report,
      });
      Office.context.document.settings.saveAsync(function (res) {
        // settings.saveAsync only writes the bag INTO the document; per MS docs
        // it reaches disk (and survives a reopen) only when the FILE is saved.
        // Persist the file so the association sticks across sessions.
        if (res && res.status === Office.AsyncResultStatus.Succeeded) {
          setDocState("Saved to this document.", "ok");
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

  function restoreSavedScan() {
    try {
      var saved = Office.context.document.settings.get(DOC_SETTING);
      if (saved && saved.report) {
        renderReport(saved.report, {
          lowConfidence: saved.lowConfidence,
          restored: true,
          savedAt: formatTs(saved.ts),
          docName: saved.docName,
        });
      } else {
        setDocState("No saved scan found in this document yet.", null);
      }
    } catch (e) {
      setDocState("Couldn't read saved scan: " + (e && e.message), "err");
    }
  }

  function formatTs(iso) {
    try { return new Date(iso).toLocaleString(); } catch (e) { return ""; }
  }

  // ── tiny helpers ───────────────────────────────────────────────────────────
  function show(el) { if (el) el.hidden = false; }
  function hide(el) { if (el) el.hidden = true; }
  function busy(on) { scanning = on; els.scanBtn.disabled = on; }
  function setStatus(msg) { els.status.textContent = msg || ""; }
  function clearResult() { els.result.innerHTML = ""; hide(els.result); setStatus(""); }
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
