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
  });

  function cache() {
    [
      "setup", "apiKey", "saveKey", "setupError",
      "scanner", "keyPrefix", "changeKey", "scanBtn", "status", "result",
      "selectionPreview", "wordCount",
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
    getSelectedText()
      .then(function (text) {
        if (!text || !text.trim()) {
          setStatus("Highlight some text in the document first, then scan.");
          return null;
        }
        return submitScan(text.trim());
      })
      .catch(function (err) { setStatus(errorText(err)); });
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
      body: JSON.stringify({ text: text }),
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
          renderResult(data, lowConfidence);
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
  function renderResult(data, lowConfidence) {
    setStatus("");
    var tier = data.tier || "unknown";
    var ai = data.ai_score == null ? "—" : Math.round(data.ai_score) + "%";
    var writing = data.writing_score == null ? "—" : Math.round(data.writing_score) + "%";

    var html =
      '<div class="dp-tier dp-tier-' + escapeAttr(tier) + '">' + escapeHtml(tierLabel(tier)) + "</div>" +
      '<dl class="dp-metrics">' +
        "<dt>AI-likelihood</dt><dd>" + ai + "</dd>" +
        "<dt>Writing quality</dt><dd>" + writing + "</dd>" +
      "</dl>";
    if (lowConfidence) {
      html += '<p class="dp-lowconf">Short selection — this read is indicative only. Scan a fuller passage (100+ words) for a confident result.</p>';
    }
    els.result.innerHTML = html;
    show(els.result);
  }

  function tierLabel(tier) {
    switch (tier) {
      case "clean": return "Clean";
      case "acceptable": return "Acceptable";
      case "concerning": return "Concerning";
      case "strong": return "Strong AI signal";
      default: return tier;
    }
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
