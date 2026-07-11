export const report = {
  "loading": "正在分析你的报告...",
  "notFound": "未找到报告。",
  "loadFailed": "无法加载报告",
  "back": "返回报告",
  "eyebrow": "分析报告",
  "eyebrowRewritten": "改写稿分析",
  "rewrittenHeroNote": "基于改写稿测量——通常仍会被标记。这是供审阅的草稿，并非干净或最终结论。",
  "documentTitle": "内容分析",
  "words": "{{count}} 词",
  "downloadPdf": "下载 PDF",
  "downloadRewrittenPdf": "下载改写稿 PDF",
  "retentionNotice": "报告会在 DraftProof 中保留 3 天。PDF 副本也会发送到你的邮箱。",
  "overview": "报告概览",
  "originalScanSummary": "原始扫描摘要",
  "originalScanBaseline": "原始扫描基线",
  // allow-hardcode: i18n UI 文案（按 code 索引的轴/等级/原因），非评分/匹配逻辑，
  // 从不与文档文本比对。与 detect.submission_risk 展示映射 + en/report.js 保持同步。
  "submissionRisk": {
    "ariaLabel": "提交风险",
    "kicker": "提交风险",
    "mainReasonPrefix": "主要原因：",
    "selfDeclare": "未知 — 需自行声明",
    "externalTrigger": "AI 可能性 ~{{score}}% — 这不是 Turnitin 分数，请勿与 20% 线比较。检测器会过度标记流畅文字，因此这是提示，并非定论。",
    "note": "这关乎你能否把它作为自己的作品站得住脚 — 而不是它看起来是否像 AI 所写。AI 使用声明、课程政策和小组分工都不在文本中，只有你才能声明。",
    "compactNote": "AI 可能性 ~{{score}}%（综合检测器）—— 这不是 Turnitin 分数；检测器会过度标记流畅文字，因此这是提示而非定论。下方「作者身份清晰度细分」中另有一项测试版深度扫描估计，不同检测器的结果可能不一致。AI 使用声明、课程政策和小组分工都不在文本中——只有你才能声明。",
    "compactNoteFused": "AI 可能性 ~{{score}}%（融合：综合 + 深度扫描检测器）—— 这不是 Turnitin 分数；检测器会过度标记流畅文字，因此这是提示而非定论。深度扫描详情见下方「作者身份清晰度细分」。AI 使用声明、课程政策和小组分工都不在文本中——只有你才能声明。",
    "compactNoteFusedAnchored": "AI 可能性 ~{{score}}%（融合：综合 + 深度扫描检测器）—— DraftProof 的标记线为 {{flagLine}}，且这不是 Turnitin 分数：不要与 Turnitin 的 20% 线比较。检测器会过度标记流畅文字，因此这是提示而非定论。深度扫描详情见下方「作者身份清晰度细分」。AI 使用声明、课程政策和小组分工都不在文本中——只有你才能声明。",
    "ownershipLead": {
      "low": "你可以把它作为自己的作品站得住脚。",
      "medium": "在能够完全将其作为自己的作品之前，请先加强它。",
      "high": "在能够完全将其作为自己的作品之前，请先加强它。"
    },
    "detectorWarning": "提醒：AI 检测器会过度标记流畅文字，因此即使这是你自己的作品，它们仍可能标记它——这是提醒，并非定论。你的保障是让内容有据可依并保留草稿。",
    "levels": {
      "low": "低",
      "medium": "中",
      "high": "高",
      "unknown": "未知"
    },
    "axes": {
      "text_pattern": "文本模式风险",
      "ownership": "归属风险",
      "citation": "引用风险",
      "defence_readiness": "答辩准备风险",
      "policy_declaration": "政策 / 声明风险"
    },
    "reasons": {
      "ownership": "对思考的归属薄弱 — 判断、推理或依据不足",
      "citation": "论断未清晰对应到来源",
      "defence_readiness": "在面谈中难以作为自己的作品来辩护",
      "text_pattern": "类似 AI 的文本模式，可能触发检测器"
    },
    // allow-hardcode: DraftProof 分数说明表的 i18n UI 文案（来自 poc/detect_v7/weights.json
    // tier_authority._notes / display_bands 的真实 FPR 数字，人工撰写的说明文字，非评分/匹配逻辑）。
    "scale": {
      "toggle": "这些数字是什么意思？",
      "headers": {
        "score": "DraftProof 分数",
        "reads": "解读",
        "measured": "我们的实测结果"
      },
      "rows": {
        "low": {
          "reads": "低",
          "measured": "约 6% 或更少的真实 ESL 学生会得到这么高的分数 — 大多数人类写作远低于此"
        },
        "medium": {
          "reads": "中",
          "measured": "真实人类写作中罕见（实测假阳性率 0.4%）"
        },
        "high": {
          "reads": "高",
          "measured": "真实人类写作中很少见（实测假阳性率低于 1%）"
        },
        "critical": {
          "reads": "严重",
          "measured": "在我们的测试中几乎从未在真实人类写作中出现（实测假阳性率 0%）"
        }
      },
      "notTurnitinComparable": "DraftProof 的分数与 Turnitin 的百分比不可比较 — 两者衡量的是不同的东西，使用不同的量表。"
    }
  },
  "repairSummary": {
    "ariaLabel": "修订摘要",
    "kicker": "修订摘要",
    "mainRisk": "主要风险",
    "statusFallback": "需要审阅",
    "mainRiskFallback": "提交前请审阅高亮段落。",
    "nextActionHighlighted": "先修复 {{count}} 个高亮段落，然后重新扫描编辑后的草稿。",
    "nextActionFallback": "审阅草稿，在需要处加入具体依据，然后重新扫描。",
    "confidenceNote": "DraftProof 报告的是写作信号，不是判定。请把它作为提交前的修订计划。",
    "editDraftHint": "重新扫描前，先编辑高亮段落。"
  },
  "groundingDiagnosis": {
    "primaryDriver": "主要问题",
    "drivers": {
      "concrete_grounding": { "label": "缺乏具体支撑", "action": "补充具体的事例、可核实的证据与细节。" },
      "authorship_trace": { "label": "作者痕迹不足", "action": "展示你的思考过程与草稿轨迹。" },
      "llm_patterning": { "label": "AI 写作模式", "action": "调整结构与措辞，避免套路化。" },
      "language_texture": { "label": "语言质感", "action": "删减空泛内容，加入具体含义。" }
    },
    // allow-hardcode: i18n 柱状标签（按 bucket code，展示用），从不与文档文本比对。
    // 与 render.py _DIAG_BUCKET_LABELS 保持同步。
    "buckets": {
      "concrete_grounding": "支撑缺口",
      "authorship_trace": "作者归属不确定",
      "llm_patterning": "类似 AI 的模式",
      "language_texture": "笼统的语言质感"
    },
    "bucketsHeading": "风险来源",
    "lowerIsBetter": "越低越好",
    "tentative": "文本较短 — 诊断仅供参考"
  },
  // allow-hardcode: i18n UI 文案（按编码索引的等级标签与指导语），非评分/匹配逻辑，
  // 与上方 groundingDiagnosis 同类，绝不与文档正文比对。
  "criticalThinking": {
    "title": "批判性思维",
    "questionsTitle": "深化思考的提问",
    "questionsIntro": "这些问题针对你自己的草稿。修改时用它们推进你的思考——答案由你自己来写。",
    "leadPrefix": "可改进",
    "strongHeading": "学生主导充分",
    "scoreLabel": "主导度评分",
    "tentative": "文本较短 — 诊断仅供参考",
    "bands": {
      "strong_control": "学生主导充分",
      "acceptable_control": "主导度尚可",
      "weak_control": "主导度偏弱",
      "high_dependency": "高度依赖 AI 风险",
      "very_high_dependency": "极高依赖 AI 风险",
      "insufficient_data": "文本不足，无法评估"
    },
    "dimensions": {
      "specific_context": { "label": "具体情境", "action": "将论点锚定到真实的作业、案例、事例或观察。" },
      "student_judgement": { "label": "学生判断", "action": "表明立场并说明你为何作出该选择。" },
      "reasoning_trail": { "label": "推理脉络", "action": "展示你如何从证据推导到结论。" },
      "ai_dependency": { "label": "AI 依赖", "action": "质疑第一个答案，而不是原样保留。" },
      "evidence_grounding": { "label": "证据支撑", "action": "为每个论点连接来源、事例或数据。" }
    },
    "reviewFlagsTitle": "复核标记（不计入评分）",
    "llmDimensions": {
      "alternative_comparison": "多方比较",
      "reflection": "反思 / 自主性"
    },
    // allow-hardcode: i18n UI 文案 — 按 LLM 输出枚举编码索引的展示标签，绝不与正文比对。
    "highlightLabels": {
      "single_path_answer": "单一路径答案",
      "no_judgement": "缺乏判断",
      "no_reflection": "缺乏反思",
      "broad_claim": "论点空泛",
      "missing_evidence": "缺少证据",
      "over_polished_closure": "结尾过于套路"
    }
  },
  "whatToFixFirst": {
    "kicker": "修订计划",
    "title": "优先修复内容",
    "intro": "这些段落是影响报告的主要部分。用只有你才知道的具体细节（一个名字、一个数字、一次观察）来支撑每一个论点，再用你自己的语言重新表述。逐句修改建议见下方的高亮部分。",
    "kickerLow": "润色",
    "titleLow": "可选润色",
    "introLow": "这篇已经读起来像你自己的作品。若想让它更无可挑剔，用只有你才知道的具体细节（一个名字、一个数字、一次观察）来支撑这些论点。逐句说明见下方的高亮部分。",
    "paragraphFallbackTitle": "审阅这个高亮段落",
    "authorshipFallbackTitle": "加强作者身份证据",
    "authorshipFallbackBody": "加入只有你能核实的具体细节。",
    "exampleTitle": "使用更有力的例子 {{count}}",
    "reviewTitle": "审阅高亮草稿",
    "reviewBody": "打开段落说明，并加强正在影响报告的部分。",
    "openParagraph": "打开段落"
  },
  "summary": {
    "riskTier": "风险等级",
    "lowConfidence": "置信度较低——分数接近等级边界，或样本较短，因此请将此等级视为临时结果，并在依赖之前进行复核。",
    "totalFindings": "发现总数",
    "authorshipRating": "作者身份评级",
    "rawAiSignal": "原始 AI 风格信号",
    "writingScore": "写作分数"
  },
  "tiers": {
    "low": "低风险",
    "moderate": "中等风险",
    "high": "高风险",
    "green": "低风险",
    "amber": "中等风险",
    "orange": "高风险",
    "red": "严重风险"
  },
  "severities": {
    "critical": "严重",
    "high": "高",
    "medium": "中",
    "low": "低",
    "info": "信息"
  },
  "transformation": {
    "scorecard": "转换模式评分卡",
    "kicker": "写作信号模式",
    "originalVsRewritten": "原始与改写模式对比",
    "patternAnalysis": "模式分析",
    "rewrittenOutcome": "改写结果",
    "rewrittenAiSignal": "改写后 AI 信号",
    "aiSignal": "AI 信号",
    "notRated": "未评级",
    "confidence": "{{value}} 置信度",
    "rewriteComparison": "改写对比",
    "notVerdict": "不是判定",
    "originalScan": "原始扫描",
    "rewrittenScan": "改写后扫描",
    "originalPatternFallback": "模式分析",
    "rewrittenPatternFallback": "改写贡献模式",
    "originalRating": "原始评级",
    "rewrittenRating": "改写后评级",
    "verdict": {
      "stillFlagged": "检测器仍会标记",
      "riskRemains": "检测风险仍在",
      "riskLow": "检测风险较低",
      "reviewed": "改写已审阅",
      "grounded": "已增强依据的草稿——请审阅",
      "groundingDetail": "依据已增强——请审阅并用你自己的话完成"
    },
    "estimatedContribution": "估计贡献",
    "calibratedAiRisk": "校准 AI 风险 {{value}}%",
    "humanAnchorDiscount": "人类锚点折减 {{value}}%",
    "calibrationConfidence": "校准置信度 {{value}}%",
    "reportingSuppression": "报告抑制 {{value}}%",
    "smartSignalsLabel": "分数信号仪表",
    "smartSignalsValue": "{{label}} {{value}}%",
    "smartSignals": {
      "aiRisk": "AI 风险",
      "humanAnchor": "人类锚点",
      "confidence": "置信度",
      "suppression": "抑制"
    },
    "contributionEstimate": "{{variant}}人类贡献与 AI 转换估计",
    "original": "原始",
    "rewritten": "改写后",
    "humanContribution": "人类贡献",
    "aiTransformation": "AI 转换",
    "signalProfile": "信号画像",
    "signals_one": "{{count}} 个信号",
    "signals_other": "{{count}} 个信号",
    "improvedFromTo": "从 {{from}}% 改善到 {{to}}%。",
    "notPresent": "此扫描中不存在该信号。",
    "summaryMixed": "混合作者身份模式：人类锚点和 AI 转换信号都可见。",
    "summaryAiDominates": "AI 转换在此扫描模式中占主导。",
    "summaryAiStronger": "AI 转换信号强于人类锚点。",
    "summaryHumanStronger": "人类贡献仍然是更强信号。",
    "mainDrivers": "主要驱动因素：{{drivers}}。",
    "rewrittenContributionEstimate": "来自已完成改写扫描的改写贡献估计。",
    "noSinglePattern": "没有单一转换模式占主导",
    "turnitinReferenceSummary": "为什么较低的外部估计会显示为 ~%",
    "turnitinReferenceNote": "Turnitin 参考：低于 20% 的 AI 分数可能显示为 *%，而不是精确百分比，因为低分区间结果可靠性较低。",
    // allow-hardcode: 转换模式 CODE 的 i18n 展示标签，从不与文档文本比对。
    // 与 poc/detect/transformation.py _LABELS 保持同步。
    "labels": {
      "fully_ai_written": "完全 AI 写作模式",
      "ai_cleaned_human_writing": "AI 润色人类写作模式",
      "ai_paraphrased": "AI 转述来源模式",
      "ai_expanded": "AI 扩展提纲模式",
      "ai_stitched_patchwork": "检测到拼接式行文",
      "ai_cited_weakly_grounded": "AI 引用但依据薄弱模式",
      "human_uncertain": "人类 / 不确定模式"
    },
    "subtitles": {
      "ai_stitched_patchwork": "部分段落像是由不同要点拼接而成。请补充衔接与来源锚点。"
    },
    "confidenceLevels": {
      "low": "低",
      "moderate": "中等",
      "high": "高"
    },
    "confidenceBadge": {
      "low": "低置信度 — 供审阅，非判定",
      "moderate": "中等置信度",
      "high": "高置信度"
    },
    "confidenceTooltip": "该信号较弱。请将其视为写作质量线索，而非作者归属判定。",
    "evidence": {
      "no_single_transformation_pattern_dominates": "没有单一转换模式占主导",
      "human_anchor_reduced_ai_certainty": "人类锚点降低了 AI 确定性"
    }
  },
  "scoreProfile": {
    "sectionLabel": "分数画像",
    "kicker": "分数画像",
    "title": "分数变化原因",
    "titleOriginal": "分数的主要驱动因素",
    "summaryRewrite": "改写后扫描会与原始扫描对比，让你看到哪些信号组已经改善，哪些仍需要审阅。",
    "summaryOriginal": "DraftProof 会把分数驱动因素分组，帮助你了解哪些信号对报告影响最大。",
    "trackedSignals": "跟踪信号",
    "noSignal": "无信号",
    "groupFallback": "用于解释分数的相关扫描器信号。",
    "leadingSignal": "主要信号 {{value}}%",
    "directionLower": "越低越好",
    "directionHigher": "越高越好",
    "improvedSignals": "{{count}} 个已改善",
    "viewDetails": "查看详细分数信号",
    "viewDetailsHint": "有改写数据时，会同时显示原始和改写后的条形。",
    "originalToRewritten": "原始 {{original}}% -> 改写后 {{rewritten}}%"
  },
  "signalGroups": {
    "ai_authorship": {
      "label": "AI 作者身份信号",
      "description": "模式、可预测性和类似模型的文本质感信号。"
    },
    "human_authenticity": {
      "label": "人类 / 真实性信号",
      "description": "会降低作者身份不确定性的依据和锚点信号。"
    },
    "quality_calibration": {
      "label": "质量与校准信号",
      "description": "证据质量、校准置信度和报告谨慎度。"
    },
    "other": {
      "label": "其他信号",
      "description": ""
    }
  },
  "signals": {
    "fallbackDescription": "用于解释转换模式的扫描器信号。",
    "labels": {
      "topk_pattern": "原始 Top-k 可预测性",
      "topk_pattern_raw": "原始 Top-k 可预测性",
      "topk_calibrated_risk": "校准 Top-k 风险",
      "ai_likelihood": "AI 可能性",
      "adjusted_ai_risk": "调整后 AI 风险",
      "calibrated_ai_risk": "校准 AI 风险",
      "human_anchor_score": "人类锚点",
      "human_anchor_discount": "人类锚点折减",
      "rewrite_smoothness": "改写平滑度",
      "semantic_uniformity_risk": "语义均匀性",
      "discourse_regularity_risk": "话语规律性",
      "source_similarity": "来源相似度",
      "surface_similarity": "表层相似度",
      "paraphrase_transformation_risk": "转述转换",
      "outline_to_text_expansion": "扩展模式",
      "section_style_variance": "拼接差异",
      "grounding_quality_risk": "依据风险",
      "citation_grounding_risk": "引用依据",
      "signal_agreement_score": "信号一致性",
      "calibration_confidence": "校准置信度",
      "reporting_suppression": "报告抑制",
      "predictability": "可预测性",
      "writing_quality": "写作质量",
      "genericity": "泛化程度",
      "ai_signal_deberta": "学习型分类器 AI 信号",
      "ai_signal_deep_scan": "待复核段落（测试版深度扫描）"
    },
    "descriptions": {
      "topk_pattern": "写作选择高度预期词语的频率。分数很高表示措辞遵循常见可预测路径。越低越好。",
      "topk_pattern_raw": "写作选择高度预期词语的频率。分数很高表示措辞遵循常见可预测路径。越低越好。",
      "topk_calibrated_risk": "DraftProof 调整原始分数后，措辞看起来有多可预测。高分可能让文本显得被机器平滑过。越低越好。",
      "ai_likelihood": "文本中检测到的整体 AI 风格写作模式。这是信号，不是作者身份证明。越低越好。",
      "adjusted_ai_risk": "DraftProof 纳入人类细节和上下文后的 AI 风格风险。越低越好。",
      "calibrated_ai_risk": "DraftProof 应用谨慎检查后的最终作者身份风险分数。越低越好。",
      "human_anchor_score": "具体的个人、地方或经验细节，让写作更像人类作者。越高越好。",
      "human_anchor_discount": "人类细节降低 AI 疑虑的程度。越高表示人类证据越强。",
      "rewrite_smoothness": "写作流畅和均匀的程度。高分可能表示文本被重度平滑。越低越好。",
      "semantic_uniformity_risk": "段落之间思想推进的均匀程度。过于均匀的节奏可能不自然。越低越好。",
      "discourse_regularity_risk": "段落流动的可预测程度。过于规律可能让写作显得模板化。越低越好。",
      "source_similarity": "含义与已知来源材料的接近程度。高分可能表示重度依赖来源。",
      "surface_similarity": "措辞与已知来源材料的接近程度。高分可能表示复制或轻度改写。",
      "paraphrase_transformation_risk": "来源含义是否被保留但措辞被大幅改变。越低越好。",
      "outline_to_text_expansion": "短想法是否以结构化方式扩展成完整写作。越低越好。",
      "section_style_variance": "各部分是否像被拼接或分多次写成。越低越好。",
      "grounding_quality_risk": "论断在多大程度上有具体、亲历、特定细节的支撑（这正是改写所改善的依据）。越低越好。与下面的引用行不同，引用行只能靠你自己的真实来源来解决。",
      "citation_grounding_risk": "论断是否有真实的引用或来源支撑。改写无法替你添加真实引用，因此在你补上自己的来源前它会保持偏高——这与改写所改善的具体、亲历细节的依据是两回事。这是完整性信号，不是 AI 证明。",
      "signal_agreement_score": "不同 DraftProof 检查指向同一方向的程度。越高表示结果越一致。",
      "calibration_confidence": "DraftProof 对可用证据足以报告的信心。越高表示置信度越高。",
      "reporting_suppression": "由于部分证据不确定或不够强，DraftProof 正在保持谨慎。越高表示应用了更多谨慎。",
      "ai_signal_deberta": "第二个学习型 AI 写作检测器的逐段分数热力图。分级：无信号（类人）、轻微、较强、高置信（>=99%）。仅供参考——高端会过度标记流利/ESL 写作，因此请复核段落，而非将颜色视为判定。"
    }
  },
  "authorship": {
    "insufficient_sample": {
      "label": "文本过短，无法评估",
      "short": "文本过短"
    },
    "ai_generated_signals": {
      "label": "AI 生成 / AI 转述信号",
      "short": "AI 信号"
    },
    "likely_ai": {
      "label": "可能 AI 辅助",
      "short": "可能 AI 辅助"
    },
    "possible_ai_assisted": {
      "label": "可能 AI 辅助",
      "short": "可能 AI 辅助"
    },
    "unlikely_ai": {
      "label": "不太可能 AI 辅助",
      "short": "不太可能 AI 辅助"
    },
    "low_ai_signal": {
      "label": "良好",
      "short": "良好"
    },
    "low_signal": {
      "label": "低 AI 信号",
      "short": "低信号"
    },
    "likely_ai_fallback": {
      "label": "可能 AI",
      "short": "可能 AI"
    }
  },
  "seal": {
    "wordsNotEnough": "{{count}} 词 · 文本不足",
    "sentencesNotEnough": "{{count}} 句 · 文本不足",
    "notEnough": "文本不足",
    "wordsSampleLimited": "{{count}} 词 · 样本有限",
    "sentencesSampleLimited": "{{count}} 句 · 样本有限",
    "sampleLimited": "样本有限",
    "calibratedTopk": "{{value}} 校准 Top-k",
    "topk": "{{value}} Top-k",
    "calibratedRisk": "{{value}} 校准风险",
    "belowReference": "低于 20% 参考线",
    "thresholdExceeded": "已超过审阅阈值",
    "rawSignal": "{{value}} 原始信号"
  },
  "aiSignalStamp": {
    "low": "低 AI 信号",
    "review": "AI 审阅",
    "likely": "可能 AI",
    "high": "高 AI 信号"
  },
  "rewrite": {
    "queued": "已排队",
    "processing": "正在改写 AI 段落",
    "retrying": "正在重试改写",
    "complete": "改写完成",
    "canceled": "改写已取消",
    "failed": "改写失败",
    "checkingFailed": "无法检查改写状态",
    "startFailed": "无法开始改写",
    "cancelFailed": "无法取消改写",
    "queuing": "正在排队改写",
    "noRewriteableTitle": "没有可自动改写的 AI 段落",
    "noRewriteableMessage": "此报告只有仅供审阅的信号。DraftProof 没有可自动改写的内容。",
    "progressMessages": [
      "正在改写被标记段落，同时保留原意。",
      "正在对照原始草稿检查修改文本。",
      "正在提升清晰度、具体性和学术语气。",
      "正在为此报告准备带高亮的改写结果。"
    ],
    "waiting": "正在等待改写 worker 启动。",
    "retryingDetail": "改写 worker 正在自动重试此步骤。",
    "checking": "仍在每隔几秒检查改写状态。",
    "longestStep": "这通常是最长的步骤。较长报告可能会停留在这里，DraftProof 正在改写并验证被标记部分。",
    "cancelTitle": "取消这次改写？",
    "cancelMessage": "DraftProof 将停止跟踪此次改写，释放保留积分，并忽略此任务稍后返回的 worker 结果。",
    "canceling": "取消中...",
    "cancelRewrite": "取消改写",
    "starting": "正在开始改写...",
    "resume": "继续改写",
    "rewriteAiSections": "自动改写 / 修正",
    "tokenEstimate": "本次改写需要 {{tokens}}，按已提交的 {{words}} 词计算。",
    "emailPdfNotice": "改写结果准备好后，我们会将 PDF 副本发送到你的账户邮箱。",
    "keepOpenInline": "请保持此报告打开；结果准备好后会显示。",
    "done": "完成",
    "elapsed": "已用时：{{elapsed}}",
    "keepOpen": "请保持此页面打开；结果会自动显示。",
    "emailPdfProgress": "改写内容的 PDF 副本也会发送到你的账户邮箱。",
    "notificationTitle": "改写完成",
    "notificationBody": "你的 DraftProof 改写结果已可查看。",
    "completion": "改写完成",
    "accepted": "依据草稿已接受",
    "topkBlocked": "Top-k 阻碍仍存在",
    "originalPreserved": "保留原文",
    "completeTitle": "改写完成",
    "acceptedDetail": "候选修改改善了目标信号，并被保留供审阅。",
    "topkBlockedDetail": "Top-k 可预测性仍然较高，因此作者应手动审阅并完成措辞。",
    "originalPreservedDetail": "没有候选修改足以改善目标信号；再次尝试前请先审阅建议。",
    "selectedDetail": "扫描后选择了一份候选修改。",
    "finishedDetail": "改写已完成，结果可以审阅。",
    "aiBefore": "改写前 AI 作者身份",
    "aiAfter": "改写后 AI 作者身份",
    "humanShift": "人类转移分数",
    "humanContribution": "改写人类贡献",
    "aiTransformation": "改写 AI 转换",
    "groundingRisk": "改写依据风险",
    "viewResult": "查看改写结果"
  },
  // allow-hardcode: i18n UI strings for the per-paragraph severity bar, not a detect/scoring word-list.
  "severityBar": {
    "caption": "全文问题密度",
    "ariaLabel": "覆盖 {{count}} 个段落的问题严重度热力图",
    "tooltip": "第 {{index}} 段：{{count}} 个问题，最严重：{{tier}}",
    "tooltipClean": "第 {{index}} 段：无问题",
    "legendLow": "干净／问题较少",
    "legendHigh": "更密集／更严重",
    "tier": {
      "critical": "严重",
      "high": "高",
      "medium": "中",
      "low": "低",
      "info": "提示"
    }
  },
  "submitted": {
    "sectionLabel": "带扫描信号的提交内容",
    "kicker": "提交内容",
    "title": "信号高亮",
    "highlightedSections": "个标记段落",
    "tabIssues": "标记段落 · {{count}}",
    "tabDocument": "查看完整文档",
    "moreDetail": "＋ 更多细节",
    "lessDetail": "− 收起细节",
    "issuesEmptyTitle": "未检测到主要问题",
    "issuesEmptyBody": "没有信号在拉高报告分数。可打开「查看完整文档」通读全文。",
    "legend": "信号颜色图例",
    "documentText": "提交文档文本",
    "selectedSignal": "所选信号详情",
    "paragraphRepairTitle": "段落修订说明",
    "signalBadge": "信号：{{value}}",
    "position": "{{current}} / {{total}} 个标记段落",
    "positionShort": "{{current}}/{{total}}",
    "editParagraph": "编辑这个段落",
    "copyGuidance": "复制建议",
    "previousIssue": "上一个问题",
    "nextIssue": "下一个问题",
    "cleanParagraphTitle": "未检测到主要问题",
    "cleanParagraphBody": "这个段落不是当前报告分数的主要来源。你仍然可以正常编辑后再重新扫描。",
    "signalStrength": "{{value}}% 信号强度",
    "signalStrengthLabel": "信号强度",
    "priority": "{{value}} 优先级",
    "paragraphSignals_one": "本段 {{count}} 个标记句子",
    "paragraphSignals_other": "本段 {{count}} 个标记句子",
    "alsoDetected": "还检测到",
    "recommendation": "如何改善这个段落",
    "criticalThinking": "批判性思维",
    "flaggedSentences": "分类器标记的句子",
    "noSignal": "没有高亮信号",
    "mapReady": "内容地图已就绪",
    "mapReadyBody": "提交文本可供审阅。新的扫描会为每个受影响段落提供更丰富的信号高亮。",
    "editor": {
      "editDraft": "手动改写 / 修正",
      "kicker": "修订工作区",
      "title": "编辑提交内容",
      "rewriteTitle": "编辑改写稿",
      "priorScanNotice": "这里显示的信号来自上一次扫描。请重新扫描编辑后的草稿以刷新发现。",
      "rewriteNotice": "您正在编辑 AI 改写稿——请将其修改为您自己的表达，然后重新扫描以刷新发现。",
      "close": "关闭",
      "noDraft": "没有本地草稿",
      "saving": "正在保存草稿...",
      "saved": "草稿已保存 {{value}}",
      "saveError": "草稿无法保存",
      "documentEditor": "可编辑文档",
      "changed": "草稿已编辑，需要重新扫描",
      "unchanged": "尚未编辑",
      "discardDraft": "丢弃草稿",
      "rescanDraft": "重新扫描编辑版本",
      "rescanning": "扫描中...",
      "emptyDraft": "请先添加文本再重新扫描。",
      "rescanQueueing": "正在排队编辑版扫描...",
      "rescanProcessing": "正在扫描编辑草稿...",
      "rescanFailed": "编辑版扫描失败。请重试。",
      "rescanTimedOut": "编辑版扫描用时较长。请稍后在报告列表中查看。",
      "translateCnEn": "中文 → English",
      "translateHint": "用中文写作？选中文字后点击「中文 → English」即可生成英文初稿。",
      "translating": "翻译中…",
      "undoTranslate": "撤销翻译",
      "translateSelectFirst": "请先选中要翻译的文字。",
      "translateError": "翻译失败，请稍后再试。",
      "translateNote": "机器翻译只是起点——在重新扫描前，请用你自己的话进行修改。",
      "trackedPreview": "修订痕迹",
      "trackedPreviewBody": "新增文本会高亮；删除文本会以删除线显示。",
      "copyTrackedChanges": "复制",
      "copiedTrackedChanges": "已复制",
      "copyTrackedChangesFailed": "复制失败",
      "amberGuideKicker": "请自行修改",
      "amberGuideHeading": "修改琥珀色句子",
      "amberGuideCopy": "这些句子被原样保留——改写在不杜撰事实的前提下无法安全改进它们，因此需要你自己来落实依据。它们在草稿中以琥珀色高亮；点击任意一句即可跳转到它，然后用你自己的话、带上具体而真实的细节重写。下方的示例演示了这种做法。",
      "amberListHeading": "原样保留——请重写这些",
      "amberItemLabel": "琥珀色句子",
      "amberJumpHint": "点击可跳转到草稿中的这句话",
      "amberResolved": "已编辑——不再被标记",
      "affectedParagraphs": "受影响段落",
      "affectedParagraph": "受影响段落",
      "paragraphUnchanged": "段落未改变",
      "paragraphEdited": "段落已编辑或移动",
      "signal": "信号"
    }
  },
  "ok": "确定",
  // allow-hardcode: 两个政策分数的 i18n UI 文案（按 code 索引），仅展示，不与文本比对。
  // 与 poc/report/render.py _POLICY_* + en/report.js 保持同步。
  "policyRisk": {
    "heading": "政策风险",
    "subheading": "在你学校的 AI 政策下，这份草稿可能如何被看待",
    "allowedLabel": "若允许使用 AI（需声明）",
    "restrictedLabel": "若不允许使用 AI",
    "mainIssuePrefix": "主要问题：",
    "bestFixPrefix": "最佳修复：",
    "confirmTitle": "自行确认以降低分数",
    "confirmAllowed": "我能说明 AI 是如何使用的",
    "confirmRestricted": "我有这份作业的草稿 / 笔记",
    "disclaimer": "这些分数并不证明使用了 AI。它们只是估计在不同学校政策下，这份草稿可能呈现多大风险。",
    "levels": {
      "low": "低", "moderate": "中等", "high": "高", "severe": "严重", "unknown": "未知"
    },
    "issues": {
      "grounding_gap": "来源支撑薄弱",
      "judgment_gap": "缺少自己的判断",
      "prompt_specificity_gap": "笼统，未紧扣任务",
      "surface_ai_text_signal": "过于光滑、类似 AI 的表层风格",
      "authorship_voice_gap": "缺少个人作者声音"
    },
    "fixes": {
      "grounding_gap": "将论断对应到真实来源与具名证据。",
      "judgment_gap": "展示你的决定、比较与评析。",
      "prompt_specificity_gap": "紧扣作业、课程模块与具体细节。",
      "surface_ai_text_signal": "变化结构与措辞，加入自己的声音。",
      "authorship_voice_gap": "加入你的个人推理过程与草稿取舍。"
    }
  },
  "aiLikelihood": {
    "title": "AI 写作信号",
    "mainFixLabel": "首先要修复的",
    // allow-hardcode: 按 tier/band code 的范围化判读展示短语（仅展示，不与文本比对）。
    // 与 render.py _SIGNAL_PHRASE / _FLAG_PHRASE 保持同步。
    "verdictSignal": {
      "green": "AI 写作信号低",
      "amber": "AI 写作信号中等",
      "orange": "AI 写作信号偏高",
      "red": "AI 写作信号偏高"
    },
    "verdictFlag": {
      "low": "在我们的标度上偏低——但检测器会过度标记流畅文字，因此仍可能标记它（这是提醒，并非定论）",
      "elevated": "可能引起外部检测器注意（并非 Turnitin 分数）",
      "high": "很可能引起外部检测器注意（这是提醒，并非定论）"
    },
    "verdictFix": "首要的写作问题是{{driver}}。",
    "draftproof": "DraftProof（保守估计）",
    "draftproofNote": "保守的 AI 风格估计——主要反映写作流畅度，加强支撑也无法完全改变它。上方的支撑修复才是真正应当着力之处。",
    "external": "Turnitin / 外部检测器（估计）",
    "externalBand": {
      "low": "不太可能被标记",
      "elevated": "可能被标记",
      "high": "很可能被标记"
    },
    "externalUnavailable": "暂无外部估计——重新扫描以生成。",
    "whyDiffer": "这是对严格第三方检测器（Turnitin、GPTZero）可能如何评分的校准估计——基于有限数据且并不精确，因为这些检测器会过度标记流畅的文字（包括真人写作）。请将其视为提示而非定论；你的保障是让内容有据可依，并用自己的话完成稿件。",
    "externalDemoted": "第三方 AI 检测器（Turnitin、GPTZero）并不完美，会过度标记流畅的文字——包括真人写作——因此预测分数并不能可靠反映你的实际结果，我们在此不再显示该分数。你的保障是让内容有据可依、用自己的话完成稿件，并保留草稿作为作者身份证据。",
    "calibrateHeading": "DraftProof 如何校准此结果"
  },
  // allow-hardcode: i18n 展示标签——按维度名称索引的界面字符串，不是评分或匹配规则
  "authenticityDashboard": {
    "ariaLabel": "真实性仪表板",
    "title": "真实性",
    "na": "—",
    "overall": "综合",
    "ciTentative": "AI 辅助估计 {{low}}–{{high}}（参考值——并非检测器定论）",
    "ciUnavailable": "无法估计 AI 辅助范围（文本不足）",
    "note": "这些维度相互重叠，仅供参考，不代表独立测量结果，亦非 Turnitin 预测。",
    "bands": { "Low": "低", "Moderate": "中等", "Medium": "中等", "High": "高" },
    "tiles": {
      "learning_ownership": "学习自主性",
      "grounding": "内容支撑",
      "citation_quality": "引用质量",
      "reasoning_consistency": "论证一致性（即将推出）",
      "ai_assistance": "AI 辅助程度"
    },
    // allow-hardcode: i18n 展示文案——按维度名称索引的提示字符串，不是评分或匹配规则
    "caveats": {
      "learning_ownership": "你在多大程度上主导思考——源自批判性思维信号。",
      "grounding": "参考值——提交内容较短。",
      "reasoning_consistency": "将在后续版本推出。"
    }
  },
  "debertaSignal": {
    "ariaLabel": "第二意见 AI 信号",
    "title": "第二意见 AI 信号",
    "scoreLabel": "的段落在另一个独立检测器下读作 AI 风格",
    "unavailable": "本次扫描无法获取第二意见检测器结果。",
    "aboveFloor": "{{pct}}% 的段落达到检测器的高置信 AI 阈值。仅供参考——事后检测器会漏掉改写文本，且存在残余的 ESL 偏差。请复核被标记的段落，而非将此数字视为定论。",
    "belowFloor": "低于我们 20% 的可靠性门槛，因此本信号不给出整体判定。我们设定了较高的标准，以避免不公平地标记流畅的 ESL 写作——标准较低的其他检测器可能会对被标记的段落（以及整篇文档）作出不同判断。请将“无判定”理解为“请复核被标记的段落”，而非“没问题”。",
    "passagesFlaggedForReview": "已标记 {{n}} / {{total}} 个段落供复核",
    "flaggedPassages": "被标记的段落",
    "reviewPassages": "请复核这些段落",
    "noteAgree": "两种检测方法都标记本文档供复核。",
    "noteDisagree": "两种检测方法对本文档的判断不同。这很常见——它们使用不同的信号。请自行复核被标记的段落。"
  },
  "authorshipBreakdown": {
    "ariaLabel": "作者归属清晰度细分",
    "title": "作者归属清晰度细分",
    "categories": {
      "student_owned": "学生原创",
      "ai_assisted_polished": "AI 辅助润色",
      "ai_paraphrased": "AI 改写",
      "ai_generated_like": "疑似 AI 生成",
      "ai_transformed": "AI 改写/生成"
    },
    "bands": {
      "Strong": "明显",
      "Some": "一定程度",
      "Little": "轻微",
      "None": "未见"
    },
    "lowConfidenceCaveat": "本文档信号不一致——请将此细分结果作为复核的起点，而非最终判定。",
    "betaChip": "测试版",
    // allow-hardcode: i18n UI copy strings (labels/band names), not a scoring or matching list.
    "fusedHeadline": {
      "label": "DraftProof AI 可能性",
      "bands": {
        "low": "低",
        "moderate": "中等",
        "high": "高",
        "critical": "严重"
      },
      "evidence": "此分数背后：综合检测器 {{composite}}%，深度扫描检测器 {{deepScan}}%（句子级）。请参见上方的 DraftProof 量表。"
    },
    "deepScan": {
      "label": "深度扫描 AI 估计",
      "insufficientChip": "证据不足",
      "notTurnitin": "句子级信号 —— 不是 Turnitin 分数",
      "bandDefersToTier": "{{band}}信号 · 总体{{tier}}",
      "bands": {
        "amber": "黄色预警",
        "orange": "橙色预警",
        "red": "红色预警"
      },
      // allow-hardcode: user-facing i18n display copy, not detection/scoring logic
      "paragraphs": {
        "head": "逐段深度扫描信号",
        "note": "同一次检测按段落分组 —— 段落越短，噪声越大",
        "row": "第 {{index}} 段",
        "sentences_other": "{{count}} 句",
        "flaggedDetail_other": "{{count}} 句中 {{flagged}} 句被标记",
        "belowFloor": "{{pct}}% —— 低于 {{floor}}% 可靠性下限，样本太少，无法判断本段",
        "atOrAboveFloor": "达到或超过 {{floor}}% 可靠性下限"
      }
    },
    "subtitle": "展示本文档的写作信号在四种作者风格之间的分布。四项占比之和恒为 100%——这是构成比例，不是 AI 概率。下方的深度扫描估计来自另一个测试版检测模型，可能与上方摘要中的「文本模式风险」不一致——两者都是信号，而非定论。",
    "uncertaintyFlags": {
      "deep_scan_uncalibrated": "包含一项仍在验证中的早期检测信号。",
      "deep_scan_below_reliability_floor": "本文档的深度扫描读数低于可靠性下限——请仅作为弱证据参考。",
      "paraphrase_without_original_draft": "缺少可对比的原始草稿，改写检测能力有限。"
    },
    "feedbackPrompt": "此细分功能处于测试阶段。它与您对文档的判断相符吗？",
    "feedbackAction": "告诉我们",
    "disclaimer": "DraftProof 提供作者归属清晰度信号与写作风险分析，并不据此判定学术不端。最终判断权属于学校、教师或相关学术政策。"
  },
  // allow-hardcode: i18n UI copy for the merged authorship+submission-risk card
  // (static labels/subtitle, not a scoring/matching oracle over document content)
  "merged": {
    "title": "作者归属与提交风险",
    "subtitle": "对本文档评分的综合解读——核心数值、写作构成分布，以及风险所在。",
    "compositionLens": "写作构成解读",
    "compositionLensNote": "合计为 100%——这是对写作风格的解读性拆分，不是上方的 AI 风险判定",
    "compositionMixedSignalsCaveat": "信号存在矛盾——以下各类别占比与本文档的风险等级不一致。请将该构成解读视为不确定，并复核被标记的段落。",
    "threeWayExplainer": "改写与生成的写作在单一文档中的信号表现相同——DraftProof 将两者合并为一项「AI 改写/生成」占比，而非勉强区分。",
    "riskLens": "风险所在",
    "riskLensNote": "各自独立",
    "verdictFramingNote": "检测结论——此等级是 DraftProof 基于校准检测器给出的 AI 风险判定。下方的构成拆分仅为展示性解读，不能覆盖该判定。"
  },
  "advancedSignals": {
    "summary": "高级信号"
  }
};
