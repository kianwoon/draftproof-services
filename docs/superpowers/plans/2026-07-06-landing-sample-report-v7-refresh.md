# Landing Sample-Report V7 Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the landing page's `SampleReportPreview` (`Landing.jsx`) back in line with what the real DraftProof scan report shows today — remove the `scoreProfile` tab (feature no longer exists), refresh `aiSignal`'s content, and restyle `actionPlan`/`findings` to match the real `FixFirstChecklist`/`SignalHighlights` components.

**Architecture:** Four independent, self-contained edits to the same two files (`Landing.jsx` + `i18n/{en,zh}/landing.js`), each one tab. Every task reuses REAL, already-shipped classes/copy from the report page/components — no new CSS, no invented vocabulary.

**Tech Stack:** React 18 + react-i18next, Vite. No test framework for this frontend — verification is `npm run build:client` plus a preview-tool visual check.

## Global Constraints

- **Line numbers in this plan reflect the file state at plan-writing time (before Task 1 applies).** Because Tasks 1-4 run sequentially and each removes/adds lines, later tasks' stated line numbers will drift. The "Currently: `<exact code>`" blocks shown in each step are the source of truth — find that exact text in the live file (grep or Read it fresh) rather than trusting the stated line number, especially for Tasks 2-4.

- No new CSS — reuse only classes already defined and loaded globally via `main.jsx:6` → `site-master.css`: `.fix-first-*` (`06-report-overview.css:761-1171`), `.issue-*` / `.deberta-evidence-*` (`07-report-submitted.css:1915-1956`), `.sample-section-card*` / `.sample-profile-bars` / `SampleSignalBar` (already used elsewhere in `Landing.jsx`).
- Every i18n key removed/added must be removed/added in BOTH `i18n/en/landing.js` and `i18n/zh/landing.js` — locale parity is mandatory in this codebase.
- Where new copy reuses real report vocabulary, use the EXACT real i18n key (e.g. `t('report.whatToFixFirst.title')`, `t('report.criticalThinking.dimensions.evidence_grounding.label')`) — never re-type a paraphrase of real copy into a new landing-only key.
- `criticalThinking` and `authorshipBreakdown` tabs are NOT touched by any task in this plan.
- Default active tab stays `'authorshipBreakdown'` (`SampleReportPreview`'s `useState('authorshipBreakdown')`) — unaffected by any task here.
- Illustrative sample data must be capability-level realistic (real bucket/severity/dimension vocabulary), not literally identical to any specific real user's data — matches the existing convention already in this file.

---

### Task 1: Remove the `scoreProfile` tab

**Files:**
- Modify: `draftproof-frontend/src/i18n/en/landing.js:405-436` (tab list), `:489-514` (orphaned keys)
- Modify: `draftproof-frontend/src/i18n/zh/landing.js:403-434` (tab list), `:487-512` (orphaned keys)
- Modify: `draftproof-frontend/src/pages/Landing.jsx:960-982` (JSX branch)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing consumed by later tasks — Tasks 2-4 touch different tabs and don't depend on tab-list ordering.

- [ ] **Step 1: Remove the `scoreProfile` entry from `reportPreviewTabs` in `i18n/en/landing.js`**

Currently (`en/landing.js:405-436`):
```js
  "reportPreviewTabs": [
    {
      "id": "authorshipBreakdown",
      "label": "Authorship Breakdown",
      "summary": "4-way composition"
    },
    {
      "id": "aiSignal",
      "label": "AI Signal",
      "summary": "Authorship pattern"
    },
    {
      "id": "scoreProfile",
      "label": "Score Profile",
      "summary": "Why it moved"
    },
    {
      "id": "actionPlan",
      "label": "Action Plan",
      "summary": "What to fix"
    },
    {
      "id": "findings",
      "label": "Findings",
      "summary": "Paragraph detail"
    },
    {
      "id": "criticalThinking",
      "label": "Critical Thinking",
      "summary": "Sharpen your thinking"
    }
  ],
```
Replace with (the `scoreProfile` object deleted, everything else unchanged):
```js
  "reportPreviewTabs": [
    {
      "id": "authorshipBreakdown",
      "label": "Authorship Breakdown",
      "summary": "4-way composition"
    },
    {
      "id": "aiSignal",
      "label": "AI Signal",
      "summary": "Authorship pattern"
    },
    {
      "id": "actionPlan",
      "label": "Action Plan",
      "summary": "What to fix"
    },
    {
      "id": "findings",
      "label": "Findings",
      "summary": "Paragraph detail"
    },
    {
      "id": "criticalThinking",
      "label": "Critical Thinking",
      "summary": "Sharpen your thinking"
    }
  ],
```

- [ ] **Step 2: Same removal in `i18n/zh/landing.js` (`zh/landing.js:403-434`)**

Currently:
```js
  "reportPreviewTabs": [
    {
      "id": "authorshipBreakdown",
      "label": "作者身份细分",
      "summary": "四类构成"
    },
    {
      "id": "aiSignal",
      "label": "AI 信号",
      "summary": "作者身份模式"
    },
    {
      "id": "scoreProfile",
      "label": "分数画像",
      "summary": "为什么变化"
    },
    {
      "id": "actionPlan",
      "label": "行动计划",
      "summary": "应该修复什么"
    },
    {
      "id": "findings",
      "label": "检测结果",
      "summary": "段落详情"
    },
    {
      "id": "criticalThinking",
      "label": "批判性思维",
      "summary": "深化你的思考"
    }
  ],
```
Replace with:
```js
  "reportPreviewTabs": [
    {
      "id": "authorshipBreakdown",
      "label": "作者身份细分",
      "summary": "四类构成"
    },
    {
      "id": "aiSignal",
      "label": "AI 信号",
      "summary": "作者身份模式"
    },
    {
      "id": "actionPlan",
      "label": "行动计划",
      "summary": "应该修复什么"
    },
    {
      "id": "findings",
      "label": "检测结果",
      "summary": "段落详情"
    },
    {
      "id": "criticalThinking",
      "label": "批判性思维",
      "summary": "深化你的思考"
    }
  ],
```

- [ ] **Step 3: Delete the now-orphaned `scoreProfile`-only keys from `i18n/en/landing.js`**

Delete these lines entirely (`en/landing.js:489-514`):
```js
  "scoreProfile": "Score Profile",
  "whyScoreMoved": "Why the score moved",
  "scoreProfileBody": "DraftProof groups scanner signals so you can see whether the score came from AI-like texture, weak grounding, or stronger human anchors.",
  "aiStyleSignal": "AI-style signal",
  "sourceGroundingSignal": "Source grounding",
  "humanAnchorSignal": "Human anchor",
  "sampleScoreSignals": [
    {
      "label": "AI-style risk",
      "value": "Low",
      "detail": "Calibrated after human anchors",
      "tone": "warning"
    },
    {
      "label": "Source grounding",
      "value": "Review",
      "detail": "Claims checked for support",
      "tone": "quality"
    },
    {
      "label": "Fix priority",
      "value": "Clear",
      "detail": "Highest-impact issues first",
      "tone": "positive"
    }
  ],
```

- [ ] **Step 4: Delete the same orphaned keys from `i18n/zh/landing.js:487-512`**

Delete these lines entirely:
```js
  "scoreProfile": "分数画像",
  "whyScoreMoved": "分数变化原因",
  "scoreProfileBody": "DraftProof 会把扫描信号分组，让你看到分数来自 AI 风格文本、来源支撑薄弱，还是更强的人类锚点。",
  "aiStyleSignal": "AI 风格信号",
  "sourceGroundingSignal": "来源支撑",
  "humanAnchorSignal": "人类锚点",
  "sampleScoreSignals": [
    {
      "label": "AI 风格风险",
      "value": "低",
      "detail": "结合人类锚点后校准",
      "tone": "warning"
    },
    {
      "label": "来源支撑",
      "value": "需审阅",
      "detail": "检查论断是否有依据",
      "tone": "quality"
    },
    {
      "label": "修复优先级",
      "value": "清晰",
      "detail": "优先处理高影响问题",
      "tone": "positive"
    }
  ],
```

- [ ] **Step 5: Remove the `scoreProfile` JSX branch from `Landing.jsx`**

Delete this entire block (`Landing.jsx:960-982`):
```jsx
        {activeSection === 'scoreProfile' && (
          <div className="sample-section-card">
            <div className="sample-section-card-head">
              <span>{t('landing.scoreProfile')}</span>
              <h3>{t('landing.whyScoreMoved')}</h3>
              <p>{t('landing.scoreProfileBody')}</p>
            </div>
            <div className="sample-score-profile-grid">
              {sampleScoreSignals.map((signal) => (
                <div className={`sample-score-signal is-${signal.tone}`} key={signal.label}>
                  <span>{signal.label}</span>
                  <strong>{signal.value}</strong>
                  <em>{signal.detail}</em>
                </div>
              ))}
            </div>
            <div className="sample-profile-bars">
              <SampleSignalBar label={t('landing.aiStyleSignal')} value={18} tone="ai" />
              <SampleSignalBar label={t('landing.sourceGroundingSignal')} value={64} tone="quality" />
              <SampleSignalBar label={t('landing.humanAnchorSignal')} value={82} tone="human" />
            </div>
          </div>
        )}

```
(Delete the entire block including the blank line that follows it, so the `actionPlan` branch that follows is not preceded by two blank lines.)

- [ ] **Step 6: Remove the now-unused `sampleScoreSignals` destructure in `Landing.jsx`'s `SampleReportPreview`**

Find this line near the top of `SampleReportPreview` (`Landing.jsx:836`):
```jsx
  const sampleScoreSignals = t('landing.sampleScoreSignals', { returnObjects: true });
```
Delete it entirely.

- [ ] **Step 7: Build check**

Run: `cd draftproof-frontend && npm run build:client`
Expected: build completes with no errors.

- [ ] **Step 8: Commit**

```bash
git add draftproof-frontend/src/i18n/en/landing.js draftproof-frontend/src/i18n/zh/landing.js draftproof-frontend/src/pages/Landing.jsx
git commit -m "fix(landing): remove scoreProfile sample tab (feature no longer exists in real report)"
```

---

### Task 2: Refresh the `aiSignal` tab content

**Files:**
- Modify: `draftproof-frontend/src/i18n/en/landing.js` (new keys + delete orphaned keys)
- Modify: `draftproof-frontend/src/i18n/zh/landing.js` (same)
- Modify: `draftproof-frontend/src/pages/Landing.jsx` (JSX inside the `aiSignal` branch)

**Interfaces:**
- Consumes: nothing from Task 1 (different tab; line numbers below assume Task 1 is already applied — the `scoreProfile` block and its keys are gone).
- Produces: nothing consumed by Tasks 3-4.

- [ ] **Step 1: Add new i18n keys to `i18n/en/landing.js`**

Add these new keys anywhere in the file's top-level object (placing them right after the existing `"reportPreviewTabs"` array, before `"sampleCriticalQuestions"`, keeps related sample-report keys grouped):
```js
  "sampleVerdictCaption": "AI-writing signal",
  "sampleVerdictLine": "Low AI-writing signal — low on our scale — but detectors over-flag fluent writing, so they may still flag it (a warning, not a verdict).",
  "sampleMainFixLabel": "Main thing to fix",
  "sampleMainFixDriver": "Grounding gap",
  "sampleMainFixAction": "Add concrete anchors, named evidence, and specifics.",
  "sampleRiskContributorsHeading": "Risk contributors",
  "sampleLowerIsBetter": "Lower is better",
  // allow-hardcode: illustrative sample-report bar values shown on the landing page —
  // fixed marketing example, never compared against document content, not a scoring oracle.
  "sampleGroundingBuckets": [
    { "label": "Grounding gap", "value": 58 },
    { "label": "Authorship uncertainty", "value": 34 },
    { "label": "AI-like patterning", "value": 22 },
    { "label": "Generic language texture", "value": 15 }
  ],
```

- [ ] **Step 2: Add the equivalent zh keys to `i18n/zh/landing.js`** (same placement, right after `reportPreviewTabs`)

```js
  "sampleVerdictCaption": "AI 写作信号",
  "sampleVerdictLine": "AI 写作信号低——在我们的量表上属于低风险——但检测工具容易误判流畅的写作，因此仍可能被标记（这是一个提醒，不是判定）。",
  "sampleMainFixLabel": "首要修复项",
  "sampleMainFixDriver": "依据缺口",
  "sampleMainFixAction": "补充具体的依据、指名的证据和细节。",
  "sampleRiskContributorsHeading": "风险构成因素",
  "sampleLowerIsBetter": "数值越低越好",
  // allow-hardcode: illustrative sample-report bar values shown on the landing page —
  // fixed marketing example, never compared against document content, not a scoring oracle.
  "sampleGroundingBuckets": [
    { "label": "依据缺口", "value": 58 },
    { "label": "作者身份不确定性", "value": 34 },
    { "label": "AI 式行文模式", "value": 22 },
    { "label": "泛化语言质感", "value": 15 }
  ],
```

- [ ] **Step 3: Delete the now-orphaned "old contribution dashboard" keys from `i18n/en/landing.js`**

Delete these lines entirely (they only existed to support the JSX being removed in Step 5):
```js
  "transformationPattern": "Transformation Pattern",
  "humanUncertain": "Human / uncertain pattern",
  "lowConfidence": "Low Confidence",
  "notVerdict": "Not a Verdict",
  "aiSignal": "AI Signal",
  "lowAiSignal": "Low AI-writing signal",
  "calibratedTopk": "41% calibrated top-k · below 20% reference",
  "originalScan": "Original Scan",
  "originalScanScore": "18%",
  "calibratedAiRisk": "Calibrated AI risk 15%",
  "humanAnchorDiscount": "Human anchor discount 38%",
  "calibrationConfidence": "Calibration confidence 61%",
  "reportingSuppression": "Reporting suppression 39%",
  "turnitinReference": "Turnitin reference: AI scores below 20% may appear as *% instead of an exact percentage because low-range results are less reliable. DraftProof scores are review signals, not verdicts.",
  "authorshipRating": "Authorship Rating",
  "good": "GOOD",
  "calibratedRisk": "11% calibrated risk",
  "estimatedContribution": "Estimated Contribution",
  "contributionBody": "Human anchoring dominates, with limited AI transformation signal.",
  "humanContribution": "Human Contribution",
  "aiTransformation": "AI Transformation",
```
(`authorshipRating`, `good`, `calibratedRisk` were already orphaned before this change — confirmed via repo-wide grep for `landing.authorshipRating`/`landing.good`/`landing.calibratedRisk`, zero references anywhere in `Landing.jsx` — safe to remove alongside this cleanup.)

Also delete, later in the file, the now-unused `sampleReportNotes` array:
```js
  "sampleReportNotes": [
    "No single transformation pattern dominates",
    "Human anchor reduced AI certainty",
    "PDF report included"
  ],
```

- [ ] **Step 4: Delete the same orphaned keys from `i18n/zh/landing.js`**

```js
  "transformationPattern": "转换模式",
  "humanUncertain": "人类 / 不确定模式",
  "lowConfidence": "低置信度",
  "notVerdict": "不是判定",
  "aiSignal": "AI 信号",
  "lowAiSignal": "AI 写作信号低",
  "calibratedTopk": "41% 校准 top-k · 低于 20% 参考线",
  "originalScan": "原始扫描",
  "originalScanScore": "18%",
  "calibratedAiRisk": "校准 AI 风险 15%",
  "humanAnchorDiscount": "人类锚点折扣 38%",
  "calibrationConfidence": "校准置信度 61%",
  "reportingSuppression": "报告抑制 39%",
  "turnitinReference": "Turnitin 参考：低于 20% 的 AI 分数可能显示为 *%，而不是精确百分比，因为低分区间结果可靠性较低。DraftProof 分数仅供审阅，不是判定。",
  "authorshipRating": "作者身份评级",
  "good": "良好",
  "calibratedRisk": "11% 校准风险",
  "estimatedContribution": "估计贡献",
  "contributionBody": "人类锚点占主导，AI 转换信号有限。",
  "humanContribution": "人类贡献",
  "aiTransformation": "AI 转换",
```
And:
```js
  "sampleReportNotes": [
    "没有单一转换模式占主导",
    "人类锚点降低了 AI 不确定性",
    "包含 PDF 报告"
  ],
```

- [ ] **Step 5: Replace the `aiSignal` tab's JSX in `Landing.jsx`**

Currently, the `aiSignal` branch (`Landing.jsx:896-958`) is:
```jsx
        {activeSection === 'aiSignal' && (
          <>
            <SubmissionRiskBand t={t} sr={SAMPLE_SUBMISSION_RISK} />
            <PolicyRiskView t={t} pr={SAMPLE_POLICY_RISK} />
            <div className="sample-report-pattern">
              <div className="sample-report-pattern-main">
                <div className="sample-transformation-icon" aria-hidden="true">
                  <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
                    <path d="M6 8.5h12.5M6 15h18M6 21.5h10" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
                    <path d="M21 7l3 3-3 3M18 18l-3 3 3 3" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <div>
                  <span>{t('landing.transformationPattern')}</span>
                  <h3>{t('landing.humanUncertain')}</h3>
                  <div className="sample-report-tags">
                    <em>{t('landing.lowConfidence')}</em>
                    <em>{t('landing.notVerdict')}</em>
                  </div>
                </div>
              </div>
              <div className="sample-authorship-badge">
                <span>{t('landing.aiSignal')}</span>
                <strong>{t('landing.lowAiSignal')}</strong>
                <small>{t('landing.calibratedTopk')}</small>
              </div>
            </div>

            <div className="sample-report-chart">
              <div className="sample-original-scan">
                <div className="sample-original-head">
                  <div>
                    <span>{t('landing.originalScan')}</span>
                    <strong>{t('landing.humanUncertain')}</strong>
                  </div>
                  <em>{t('landing.originalScanScore')}</em>
                </div>

                <div className="sample-contribution">
                  <span>{t('landing.estimatedContribution')}</span>
                  <p>{t('landing.contributionBody')}</p>
                  <div className="sample-report-tags">
                    <em>{t('landing.calibratedAiRisk')}</em>
                    <em>{t('landing.humanAnchorDiscount')}</em>
                    <em>{t('landing.calibrationConfidence')}</em>
                    <em>{t('landing.reportingSuppression')}</em>
                  </div>
                  <div className="sample-contribution-bars">
                    <SampleSignalBar label={t('landing.humanContribution')} value={100} tone="human" />
                    <SampleSignalBar label={t('landing.aiTransformation')} value={0} tone="ai" />
                  </div>
                </div>
              </div>

              <div className="sample-report-notes">
                {sampleReportNotes.map((note) => (
                  <span key={note}>{note}</span>
                ))}
              </div>
              <p className="sample-reference-note">{t('landing.turnitinReference')}</p>
            </div>
          </>
        )}
```
Replace it with:
```jsx
        {activeSection === 'aiSignal' && (
          <>
            <SubmissionRiskBand t={t} sr={SAMPLE_SUBMISSION_RISK} />
            <PolicyRiskView t={t} pr={SAMPLE_POLICY_RISK} />
            <div className="sample-section-card">
              <div className="sample-section-card-head">
                <span>{t('landing.sampleVerdictCaption')}</span>
                <h3>{t('landing.sampleVerdictLine')}</h3>
              </div>
              <p>
                <strong>{t('landing.sampleMainFixLabel')}: {t('landing.sampleMainFixDriver')}</strong>
                {' — '}
                {t('landing.sampleMainFixAction')}
              </p>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '.04em', margin: '12px 0 6px' }}>
                <span>{t('landing.sampleRiskContributorsHeading')}</span>
                <span>{t('landing.sampleLowerIsBetter')}</span>
              </div>
              <div className="sample-profile-bars">
                {sampleGroundingBuckets.map((bucket) => (
                  <SampleSignalBar key={bucket.label} label={bucket.label} value={bucket.value} tone="ai" />
                ))}
              </div>
            </div>
          </>
        )}
```

- [ ] **Step 6: Remove the now-unused `sampleReportNotes` destructure and add the new `sampleGroundingBuckets` one**

Find this line near the top of `SampleReportPreview` (`Landing.jsx:835`, or one line above wherever Task 1 Step 6 left `sampleActionItems`'s neighbor if Task 1 already ran — search for the exact text, not the line number, since Task 1's removal shifts subsequent lines up by one):
```jsx
  const sampleReportNotes = t('landing.sampleReportNotes', { returnObjects: true });
```
Replace it with:
```jsx
  const sampleGroundingBuckets = t('landing.sampleGroundingBuckets', { returnObjects: true });
```

- [ ] **Step 7: Build check**

Run: `cd draftproof-frontend && npm run build:client`
Expected: build completes with no errors.

- [ ] **Step 8: Commit**

```bash
git add draftproof-frontend/src/i18n/en/landing.js draftproof-frontend/src/i18n/zh/landing.js draftproof-frontend/src/pages/Landing.jsx
git commit -m "feat(landing): refresh aiSignal sample tab to match real verdict+grounding-diagnosis card"
```

---

### Task 3: Restyle the `actionPlan` tab to match `FixFirstChecklist`

**Files:**
- Modify: `draftproof-frontend/src/i18n/en/landing.js` (`sampleActionItems` data shape, delete 3 keys)
- Modify: `draftproof-frontend/src/i18n/zh/landing.js` (same)
- Modify: `draftproof-frontend/src/pages/Landing.jsx` (JSX inside the `actionPlan` branch)

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: nothing consumed by Task 4.

- [ ] **Step 1: Change `sampleActionItems`'s shape in `i18n/en/landing.js`** — replace `tone` with `label`

Currently:
```js
  "sampleActionItems": [
    {
      "title": "Add citation support",
      "body": "Two claims need clearer source backing before submission.",
      "tone": "warning"
    },
    {
      "title": "Strengthen source grounding",
      "body": "One paragraph should explain how the cited source supports the point.",
      "tone": "quality"
    },
    {
      "title": "Revise generic phrasing",
      "body": "Replace broad AI-style wording with specific reasoning and evidence.",
      "tone": "positive"
    }
  ],
```
Replace with:
```js
  "sampleActionItems": [
    {
      "title": "Add citation support",
      "body": "Two claims need clearer source backing before submission.",
      "label": "High priority"
    },
    {
      "title": "Strengthen source grounding",
      "body": "One paragraph should explain how the cited source supports the point.",
      "label": "Medium priority"
    },
    {
      "title": "Revise generic phrasing",
      "body": "Replace broad AI-style wording with specific reasoning and evidence.",
      "label": "Quick win"
    }
  ],
```

- [ ] **Step 2: Same shape change in `i18n/zh/landing.js`**

Currently:
```js
  "sampleActionItems": [
    {
      "title": "补充引用支撑",
      "body": "有两个论断在提交前需要更清楚的来源依据。",
      "tone": "warning"
    },
    {
      "title": "加强来源支撑",
      "body": "有一段需要说明引用来源如何支撑你的观点。",
      "tone": "quality"
    },
    {
      "title": "修改泛化表达",
      "body": "用更具体的推理和证据替换宽泛的 AI 风格措辞。",
      "tone": "positive"
    }
  ],
```
Replace with:
```js
  "sampleActionItems": [
    {
      "title": "补充引用支撑",
      "body": "有两个论断在提交前需要更清楚的来源依据。",
      "label": "高优先级"
    },
    {
      "title": "加强来源支撑",
      "body": "有一段需要说明引用来源如何支撑你的观点。",
      "label": "中优先级"
    },
    {
      "title": "修改泛化表达",
      "body": "用更具体的推理和证据替换宽泛的 AI 风格措辞。",
      "label": "快速改进项"
    }
  ],
```

- [ ] **Step 3: Delete the now-unused `actionPlan`/`actionPlanTitle`/`actionPlanBody` keys from `i18n/en/landing.js`**

```js
  "actionPlan": "Action Plan",
  "actionPlanTitle": "Fix the highest-impact issues first",
  "actionPlanBody": "The report turns scan signals into a practical review order before you revise or download the PDF.",
```
(These are replaced by the real `report.whatToFixFirst.kicker`/`.title` keys, already loaded globally and already used elsewhere in this exact file for the `criticalThinking` tab's cross-namespace reuse — same established pattern.)

- [ ] **Step 4: Delete the same 3 keys from `i18n/zh/landing.js`**

```js
  "actionPlan": "行动计划",
  "actionPlanTitle": "先修复影响最大的问题",
  "actionPlanBody": "报告会把扫描信号转化为实用的审阅顺序，帮助你在修改或下载 PDF 前知道先处理什么。",
```

- [ ] **Step 5: Replace the `actionPlan` tab's JSX in `Landing.jsx`**

Currently (`Landing.jsx:984-1003`):
```jsx
        {activeSection === 'actionPlan' && (
          <div className="sample-section-card">
            <div className="sample-section-card-head">
              <span>{t('landing.actionPlan')}</span>
              <h3>{t('landing.actionPlanTitle')}</h3>
              <p>{t('landing.actionPlanBody')}</p>
            </div>
            <div className="sample-action-list">
              {sampleActionItems.map((item, index) => (
                <article className={`sample-action-item is-${item.tone}`} key={item.title}>
                  <span>{String(index + 1).padStart(2, '0')}</span>
                  <div>
                    <strong>{item.title}</strong>
                    <p>{item.body}</p>
                  </div>
                </article>
              ))}
            </div>
          </div>
        )}
```
Replace with:
```jsx
        {activeSection === 'actionPlan' && (
          <div className="fix-first-card">
            <div className="fix-first-head">
              <span>{t('report.whatToFixFirst.kicker')}</span>
              <h2>{t('report.whatToFixFirst.title')}</h2>
            </div>
            <div className="fix-first-list">
              {sampleActionItems.map((item, index) => (
                <div className="fix-first-item" key={item.title}>
                  <span className="fix-first-index">{index + 1}</span>
                  <span className="fix-first-copy">
                    <strong>{item.title}</strong>
                    <em>{item.body}</em>
                  </span>
                  {item.label && <span className="fix-first-label">{item.label}</span>}
                </div>
              ))}
            </div>
          </div>
        )}
```

- [ ] **Step 6: Build check**

Run: `cd draftproof-frontend && npm run build:client`
Expected: build completes with no errors.

- [ ] **Step 7: Commit**

```bash
git add draftproof-frontend/src/i18n/en/landing.js draftproof-frontend/src/i18n/zh/landing.js draftproof-frontend/src/pages/Landing.jsx
git commit -m "feat(landing): restyle actionPlan sample tab to match real FixFirstChecklist"
```

---

### Task 4: Restructure the `findings` tab to match `SignalHighlights`'s issue-card

**Files:**
- Modify: `draftproof-frontend/src/i18n/en/landing.js` (new keys, delete orphaned keys)
- Modify: `draftproof-frontend/src/i18n/zh/landing.js` (same)
- Modify: `draftproof-frontend/src/pages/Landing.jsx` (JSX inside the `findings` branch)

**Interfaces:**
- Consumes: nothing from Tasks 1-3.
- Produces: nothing — final tab in this plan.

- [ ] **Step 1: Add new i18n keys to `i18n/en/landing.js`** (place near the existing `findingsSample*` keys)

```js
  "findingsSamplePosition": "2/5",
  "findingsSampleCount": "3 flagged sentences in paragraph",
  // allow-hardcode: illustrative sample flagged-sentence evidence shown on the landing
  // page — fixed marketing example built from the same sample paragraph used elsewhere
  // on this page, never compared against document content, not a scoring/matching oracle.
  "sampleFlaggedSentences": [
    {
      "text": "In addition to economics, the United States has a strong cultural influence.",
      "score": 61,
      "suggestion": "Anchor to a specific example or source instead of a general claim."
    },
    {
      "text": "The entertainment industry in Hollywood has become one of the most powerful cultural exports in history.",
      "score": 74,
      "suggestion": "Beyond its economy, American cinema reaches audiences in over 190 countries each year."
    }
  ],
```

- [ ] **Step 2: Add the equivalent zh keys to `i18n/zh/landing.js`**

```js
  "findingsSamplePosition": "2/5",
  "findingsSampleCount": "段落中有 3 个被标记的句子",
  // allow-hardcode: illustrative sample flagged-sentence evidence shown on the landing
  // page — fixed marketing example built from the same sample paragraph used elsewhere
  // on this page, never compared against document content, not a scoring/matching oracle.
  "sampleFlaggedSentences": [
    {
      "text": "除经济领域外，美国还拥有强大的文化影响力。",
      "score": 61,
      "suggestion": "应锚定到具体的例子或来源，而不是泛泛而谈的论断。"
    },
    {
      "text": "好莱坞娱乐产业已成为史上最强大的文化出口之一。",
      "score": 74,
      "suggestion": "超越经济领域，美国电影每年覆盖全球190多个国家的观众。"
    }
  ],
```

- [ ] **Step 3: Delete the now-orphaned findings keys from `i18n/en/landing.js`**

```js
  "findingsSampleId": "S004–S006",
  "findingsSignalStrength": "Signal Strength",
  "findingsSampleChip1": "8 Findings In Paragraph",
  "findingsSampleChip2": "MEDIUM Priority",
  "findingsSampleChip3": "Auto-Fixable",
  "findingsAlsoDetected": "Also Detected",
  "findingsSampleAlso": "Generic Phrasing",
  "findingsMainIssue": "Main Issue to Fix",
  "findingsSampleMainIssue": "Use of generic transitional phrase and predictable wording that reduces originality.",
  "findingsRewriteHint": "Rewrite Hint",
  "findingsSampleRewriteHint": "Example: \"Beyond its economy, American cinema reaches audiences in over 190 countries each year.\"",
```
Also delete `findingsSampleParagraph` (its content is now fully represented, split across the two `sampleFlaggedSentences` entries added in Step 1 — keeping it would leave a stale, unreferenced duplicate of the same text):
```js
  "findingsSampleParagraph": "In addition to economics, the United States has a strong cultural influence. American movies, music, fashion, and social media trends are consumed globally. The entertainment industry in Hollywood has become one of the most powerful cultural exports in history.",
```
**Keep** `findingsSampleType` ("AI Likelihood") and `findingsSampleDescription` — both are still used (Step 6 below).

- [ ] **Step 4: Delete the same orphaned keys from `i18n/zh/landing.js`**

```js
  "findingsSampleId": "S004–S006",
  "findingsSignalStrength": "信号强度",
  "findingsSampleChip1": "段落中 8 个发现",
  "findingsSampleChip2": "中等优先级",
  "findingsSampleChip3": "可自动修复",
  "findingsAlsoDetected": "同时检测到",
  "findingsSampleAlso": "通用措辞",
  "findingsMainIssue": "主要问题",
  "findingsSampleMainIssue": "使用了通用过渡短语和可预测措辞，降低了文章的原创性。",
  "findingsRewriteHint": "改写提示",
  "findingsSampleRewriteHint": "示例：\"超越经济领域，美国电影每年覆盖全球190多个国家的观众。\"",
```
And:
```js
  "findingsSampleParagraph": "除经济领域外，美国还拥有强大的文化影响力。美国的电影、音乐、时尚和社交媒体潮流在全球范围内被广泛消费。好莱坞娱乐产业已成为史上最强大的文化出口之一。",
```

- [ ] **Step 5: Replace the `findings` tab's JSX in `Landing.jsx`**

Currently (`Landing.jsx:1005-1045`):
```jsx
        {activeSection === 'findings' && (
          <div className="sample-finding-card">
            <div className="sample-finding-header">
              <div>
                <span className="sample-finding-id">{t('landing.findingsSampleId')}</span>
                <h3 className="sample-finding-type">{t('landing.findingsSampleType')}</h3>
              </div>
              <span className="sample-finding-num">#4</span>
            </div>
            <div className="sample-finding-body">
              <blockquote className="sample-finding-paragraph">
                {t('landing.findingsSampleParagraph')}
              </blockquote>
              <p className="sample-finding-description">{t('landing.findingsSampleDescription')}</p>
              <div className="sample-finding-strength-row">
                <span>{t('landing.findingsSignalStrength')}</span>
                <strong>59%</strong>
              </div>
              <div className="sample-signal-track">
                <i className="is-ai" style={{ width: '59%' }} />
              </div>
              <div className="sample-finding-chips">
                <em>{t('landing.findingsSampleChip1')}</em>
                <em>{t('landing.findingsSampleChip2')}</em>
                <em>{t('landing.findingsSampleChip3')}</em>
              </div>
              <div className="sample-finding-also">
                <span>{t('landing.findingsAlsoDetected')}</span>
                <em>{t('landing.findingsSampleAlso')}</em>
              </div>
              <div className="sample-finding-subsection">
                <span>{t('landing.findingsMainIssue')}</span>
                <p>{t('landing.findingsSampleMainIssue')}</p>
              </div>
              <div className="sample-finding-subsection">
                <span>{t('landing.findingsRewriteHint')}</span>
                <p>{t('landing.findingsSampleRewriteHint')}</p>
              </div>
            </div>
          </div>
        )}
```
Replace with:
```jsx
        {activeSection === 'findings' && (
          <div className="issue-card is-open">
            <div className="issue-card-head">
              <span className="issue-card-main">
                <span className="issue-card-chips">
                  <em className="issue-card-num">{t('landing.findingsSamplePosition')}</em>
                  <em className="issue-chip issue-chip-tier is-high">{t('report.severities.high')}</em>
                  <em className="issue-chip">{t('landing.findingsSampleType')}</em>
                  <em className="issue-chip">{t('landing.findingsSampleCount')}</em>
                </span>
              </span>
            </div>
            <div className="issue-card-body">
              <p className="issue-card-summary">{t('landing.findingsSampleDescription')}</p>
              <div className="issue-action issue-action-thinking">
                <span className="issue-action-label">{t('report.submitted.criticalThinking')}</span>
                <p>
                  <strong>{t('report.criticalThinking.dimensions.evidence_grounding.label')}</strong>
                  {' — '}
                  {t('report.criticalThinking.dimensions.evidence_grounding.action')}
                </p>
              </div>
              <div className="issue-action issue-action-evidence">
                <span className="issue-action-label">{t('report.submitted.flaggedSentences')}</span>
                <ul className="deberta-evidence-list">
                  {sampleFlaggedSentences.map((sentence) => (
                    <li key={sentence.text}>
                      <span className="deberta-evidence-score">{sentence.score}%</span>
                      <span className="deberta-evidence-text">{sentence.text}</span>
                      <span className="deberta-evidence-suggestion">{sentence.suggestion}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
```

- [ ] **Step 6: Add the `sampleFlaggedSentences` destructure in `SampleReportPreview`**

Find this line near the top of `SampleReportPreview` (`Landing.jsx:837` at plan-writing time — search for the exact text below, since Tasks 1-3 shift line numbers before this task runs):
```jsx
  const sampleActionItems = t('landing.sampleActionItems', { returnObjects: true });
```
Add immediately after it:
```jsx
  const sampleFlaggedSentences = t('landing.sampleFlaggedSentences', { returnObjects: true });
```

- [ ] **Step 7: Build check**

Run: `cd draftproof-frontend && npm run build:client`
Expected: build completes with no errors.

- [ ] **Step 8: Visual verification via preview tool (covers all 4 tasks in this plan)**

Start the dev server (reuse `.claude/launch.json` config `"frontend"` from prior work in this worktree). Then:
- Navigate to `/`, scroll to the sample-report preview: confirm exactly 5 tabs (`authorshipBreakdown`, `aiSignal`, `actionPlan`, `findings`, `criticalThinking` — no `scoreProfile`).
- `aiSignal` tab: `SubmissionRiskBand` + `PolicyRiskView` render, followed by the verdict line, "Main thing to fix" callout, and 4 grounding-bucket bars (Grounding gap highest, Generic language texture lowest) — no leftover "Estimated Contribution"/transformation-pattern content.
- `actionPlan` tab: kicker "Repair Plan", title "What to fix first", 3 numbered items each with a title/body/label chip — no colored tone styling.
- `findings` tab: position chip "2/5", tier chip "HIGH", signal chip "AI Likelihood", count chip, reader-summary paragraph, a purple-accented "Critical thinking" callout, and a 2-item flagged-sentences list with score/text/suggestion — no "also-detected/main-issue/rewrite-hint" chips.
- `criticalThinking` tab unchanged.
- Repeat on `/zh`: full Chinese localization, no English fallback text anywhere in these 4 tabs.
- Check console for errors (`mcp__Claude_Preview__preview_console_logs`, level `error`) on both locales.

- [ ] **Step 9: Commit**

```bash
git add draftproof-frontend/src/i18n/en/landing.js draftproof-frontend/src/i18n/zh/landing.js draftproof-frontend/src/pages/Landing.jsx
git commit -m "feat(landing): restructure findings sample tab to match real SignalHighlights issue-card"
```

---

## Self-Review Notes

- **Spec coverage:** scoreProfile removal → Task 1. aiSignal refresh → Task 2. actionPlan restyle → Task 3. findings restructure → Task 4. i18n cleanup (all orphaned keys named in the spec) → distributed across the task that orphans them. criticalThinking/authorshipBreakdown untouched → confirmed, no task touches them. ✓
- **Placeholder scan:** no TBD/TODO; every step has complete, copy-pasteable before/after code. ✓
- **Type/name consistency:** `sampleGroundingBuckets` (Task 2) has `{label, value}` shape, consumed identically by `SampleSignalBar` (pre-existing component, already takes `label`/`value`/`tone` props — confirmed via its existing use elsewhere in this file). `sampleActionItems`' new `label` field (Task 3) is read as `item.label` in the same step that defines it. `sampleFlaggedSentences` (Task 4) has `{text, score, suggestion}`, consumed identically in the same step. No drift between any task's data shape and its consumer. ✓
- **Cross-task file overlap:** all 4 tasks touch the same two i18n files and `Landing.jsx`, but each touches a different, non-overlapping tab's lines — tasks must be applied in order (1→2→3→4) since later tasks' line-number references assume earlier tasks already landed, but they do not depend on each other's data/interfaces.
