export const landing = {
  "rewriteTeaserEyebrow": "改写",
  "rewriteTeaserTitle": "有论断被标记了？看看该怎么修。",
  "rewriteTeaserBody": "当扫描标记出单薄的段落时，改写会给出一份可对照的「改写前／改写后」——一份你用自己的话写完的学习草稿。",
  "rewriteTeaserBeforeLabel": "改写前",
  "rewriteTeaserAfterLabel": "改写后",
  "rewriteTeaserBefore": "好莱坞已成为史上最具影响力的文化输出之一。",
  "rewriteTeaserAfter": "好莱坞是全球输出规模最大的文化产业之一〔补一个你能核实的引用〕。",
  "rewriteTeaserMarker": "待复核的补充内容——请用你自己的来源替换括号中的提示。",
  "rewriteTeaserLink": "了解改写的工作方式 →",
  "heroPill": "面向学生的依据支撑与批判性思维审阅",
  "heroTitle": "确保草稿里的思考，依然属于你。",
  "heroTitleHighlight": "依然属于你",
  "heroTitlesLabel": "轮播标题信息",
  "heroTitles": [
    { "text": "确保草稿里的思考，依然属于你。", "highlight": "依然属于你" },
    { "text": "在导师之前，先抓出你还无法支撑的论断。", "highlight": "你还无法支撑的论断" },
    { "text": "按你学校的规则来读你的草稿，而不是一个笼统的分数。", "highlight": "而不是一个笼统的分数" },
    { "text": "用你自己的证据和声音，为每个观点立稳依据。", "highlight": "你自己的证据和声音" }
  ],
  "heroLead": "DraftProof 会指出你还无法支撑的论断，以及读起来像机器写的段落，再追问你真正想表达什么——让你在提交前用自己的证据和推理，为每一个观点立稳依据。",
  "runCheck": "审阅我的内容",
  "viewSample": "查看示例报告",
  // allow-hardcode: i18n 展示文案。"5" 对应 WELCOME_CREDITS 默认值（config.py），若该环境变量变更需同步更新；非评分依据。
  // 拆分为 pre/link/post，使“注册”渲染为指向登录页的链接。
  "heroFreeCreditsPre": "新用户？",
  "heroFreeCreditsLink": "注册",
  "heroFreeCreditsPost": "即送 5 个免费积分，无需信用卡即可试用。",
  "trustNote": "不是 Turnitin 的替代品 · 不是学术不端判定 · 在提交前做好准备，并留存负责任使用 AI 的证据。",
  // allow-hardcode: i18n 展示文案，用于落地页用例轮播——仅作为界面文本呈现给用户，
  // 绝不与文档内容比对、也不作为评分/匹配判定（与上方 heroReviewSteps/anchorCards 同类）。
  "useCasesLabel": "DraftProof 如何针对你的情况提供帮助",
  "useCasesEyebrow": "围绕你学校对 AI 的真实态度而设计",
  "useCasesHeading": "不只是一个分数——针对你具体情况的指导",
  "useCasesTabsLabel": "用例分区",
  "useCasesNext": "显示下一个用例",
  "useCases": [
    {
      "id": "score-vs-coaching",
      "eyebrow": "分数不等于帮助",
      "tag": "检测工具的问题",
      "title": "一个百分比并不会告诉你该改什么。",
      "body": [
        "Turnitin 和免费 AI 检测器只丢回一个数字，剩下的让你自己猜。一个百分比只告诉你被标记了——却从不告诉你是哪一句、为什么、接下来该怎么办。",
        "DraftProof 会去读这份草稿：标出存在风险的具体论断与段落，并为每一处给出高亮的前后对照，让你在提交前看清该补充什么、该锚定到哪里、该用自己的话改写哪些内容。"
      ],
      // allow-hardcode: i18n 展示文案（轮播分页的要点/护栏/收束语，呈现给用户）——非评分/匹配判定。
      "points": [
        "精确定位存在风险的具体论断与段落",
        "解释每个段落为何读起来像机器所写",
        "提供可由你用自己内容修改的前后对照",
        "把分数转化为具体、可执行的修改步骤",
        "诚实：会告诉你哪些地方检测器仍可能标记"
      ],
      "guardrails": [
        "不是用来博弈的分数，也不是学术不端判定。",
        "一个数字只说你被标记了——DraftProof 告诉你为什么，以及怎么改。"
      ],
      "punch": "检测器只丢给你一个判定；DraftProof 给你一份可执行的修改稿。"
    },
    {
      "id": "ai-allowed",
      "eyebrow": "如果你的学校允许使用 AI",
      "tag": "正确地借助 AI",
      "title": "证明你借助 AI 完成的作业有依据、且属于你。",
      "body": [
        "在允许使用 AI 的情况下，标准不在于你是否用了它——而在于质量与掌控力。DraftProof 更看重依据、判断与具体性，对表层的 AI 用词则从轻处理。",
        "它会像一门允许使用 AI 的课程那样来读你的草稿：每个论断是否建立在真实证据与你自己的推理之上？你能清楚看到哪里依据扎实、哪里仍读起来笼统——从而把作业打磨成无可争议地属于你。"
      ],
      // allow-hardcode: i18n 展示文案（轮播分页的要点/护栏/收束语，呈现给用户）——非评分/匹配判定。
      "points": [
        "看重真实的证据、推理与你自己的表述框架",
        "检查每个论断是否锚定到来源或实例",
        "表层 AI 用词权重从轻——使用 AI 本身没问题",
        "标出笼统、缺乏支撑的论断以便立稳依据",
        "为允许且已声明的 AI 使用而调校的宽松分档"
      ],
      "guardrails": [
        "不是用来博弈的分数，也不是学术不端判定。",
        "在允许使用 AI 的地方，标准是质量与掌控力——而不是你是否用了 AI。"
      ],
      "punch": "合格的 AI 辅助作业，依据扎实、内容具体，并且依然属于你。"
    },
    {
      "id": "ai-banned",
      "eyebrow": "如果你的学校禁止使用 AI",
      "tag": "证明思考出自你",
      "title": "证明作业由你撰写，而非模型。",
      "body": [
        "在禁止使用 AI 的情况下，DraftProof 更看重作者声音与表层 AI 文本，并标出读起来像机器所写的段落——让你在读者提出质疑之前，先用自己的声音改写。",
        "它无法证明你是怎么写的，也从不假装能证明。它会标出读者可能质疑的段落，以及只有你能确认的因素——让真正站得住的防线成立：思考与文字，都出自你。绝不绕过检测，也绝不做出指控。"
      ],
      // allow-hardcode: i18n 展示文案（轮播分页的要点/护栏/收束语，呈现给用户）——非评分/匹配判定。
      "points": [
        "标出读起来不像你声音的段落",
        "更看重作者声音与表层 AI 文本",
        "指出在何处用你自己的语言与推理重写",
        "更严格的分档——为零 AI 政策而设",
        "由你掌控的自行确认因素，绝不擅自假定"
      ],
      "guardrails": [
        "不是用来博弈的分数，也不是'你用了 AI'的指控。",
        "我们无法证明你是怎么写的——只有你能，靠的是真正掌控思考。"
      ],
      "punch": "亲自主导思考，才是真正的防线，而不是去伪装文字。"
    }
  ],
  "quickSummary": "DraftProof 快速摘要",
  "livePreview": "实时预览",
  "preSubmissionReview": "内容提交前审阅",
  "runningCheck": "正在检查内容提交前风险...",
  "reviewTier": "审阅等级",
  "medium": "中等",
  "grounding": "依据支撑",
  "strong": "强",
  "citationGaps": "引用缺口",
  "foundCount": "发现 {{count}} 个",
  "sourceIntegrity": "来源完整性",
  "verified": "已验证",
  "primaryFix": "主要修复",
  "oneCitation": "1 条引用",
  "actionable": "可执行",
  "heroReviewStepsLabel": "DraftProof 审阅预览步骤",
  "heroReviewSteps": [
    {
      "id": "scan-draft",
      "visual": "scan",
      "stepLabel": "步骤 1",
      "shortLabel": "01",
      "title": "正在扫描你的草稿",
      "body": "DraftProof 会在提交前读取论断、引用、来源连接和类似 AI 的写作模式。",
      "scanLines": [
        {
          "label": "论断",
          "text": "好莱坞已经成为最强大的文化出口之一。",
          "tone": "warning"
        },
        {
          "label": "来源",
          "text": "这句话附近还没有可见的来源连接。",
          "tone": "warning"
        },
        {
          "label": "语境",
          "text": "保留学生自己的论点和课程重点。",
          "tone": "positive"
        }
      ],
      "scanStatus": [
        {
          "label": "地图",
          "value": "读取中",
          "tone": "positive"
        },
        {
          "label": "引用",
          "value": "缺失",
          "tone": "warning"
        },
        {
          "label": "语气",
          "value": "保留",
          "tone": "positive"
        }
      ],
      "primaryLabel": "当前重点",
      "primaryValue": "缺少锚点的论断",
      "badge": "扫描中"
    },
    {
      "id": "find-risks",
      "visual": "findings",
      "stepLabel": "步骤 2",
      "shortLabel": "02",
      "title": "已发现内容风险",
      "body": "审阅会把证据缺口和措辞信号分开，让你知道哪里可能被质疑。",
      "findings": [
        {
          "label": "引用缺口",
          "body": "宽泛论断需要附近的引用或来源锚点。",
          "badge": "高",
          "tone": "warning"
        },
        {
          "label": "依据薄弱",
          "body": "读者还看不清来源和句子的关系。",
          "badge": "中",
          "tone": "warning"
        },
        {
          "label": "来源完整性",
          "body": "现有来源信息可用，只需要放到正确位置。",
          "badge": "已验证",
          "tone": "positive"
        }
      ],
      "summary": [
        {
          "label": "审阅等级 · 中等",
          "tone": "warning"
        },
        {
          "label": "2 个引用风险",
          "tone": "warning"
        },
        {
          "label": "来源已验证",
          "tone": "positive"
        }
      ],
      "primaryLabel": "主要风险",
      "primaryValue": "证据缺口",
      "badge": "已发现"
    },
    {
      "id": "fix-plan",
      "visual": "diff",
      "stepLabel": "步骤 3",
      "shortLabel": "03",
      "title": "修改计划已生成",
      "body": "DraftProof 会把发现转成清楚的修复顺序，让你在提交前直接行动。",
      "diffRows": [
        {
          "label": "修改前",
          "text": "好莱坞已经成为历史上最强大的文化出口之一。",
          "tone": "remove"
        },
        {
          "label": "修改后",
          "text": "在来源支持下，好莱坞通过电影、流媒体和音乐影响全球受众。",
          "tone": "add"
        }
      ],
      "fixSteps": [
        "补上来源",
        "连接来源与论断",
        "保留原意"
      ],
      "primaryLabel": "主要修复",
      "primaryValue": "为论断补上来源",
      "badge": "可执行"
    }
  ],
  "builtFor": "适用于",
  "audienceDetails": "DraftProof 适用人群和审阅详情",
  "students": "本科生",
  "gradWriters": "研究生与论文写作者",
  "eslWriters": "非母语英语写作者",
  "independentWriters": "独立研究者",
  "tokenRate": "每 1,000 词 1 个积分",
  "pdfReport": "PDF 报告",
  "citationGrounding": "引用 + 相似度审阅",
  "contentAwareRewrite": "依据感知修改建议",
  "contentCarouselLabel": "DraftProof 依据支撑与 AI 风险背景",
  "contentCarouselTabsLabel": "依据背景分区",
  "contentCarouselNext": "显示下一个依据背景分区",
  "humanWrittenEyebrow": "类似 AI 的信号",
  "humanWrittenTitle": "人类亲自写的内容，也可能看起来像 AI 生成。",
  "humanWrittenBody1": "AI 写作报告并不知道每一句是谁写的。它读取的是模式——可预测性、句子节奏、泛化表达、薄弱依据——这些即便在人写的作品里也可能像 AI 生成的文本。",
  "humanWrittenBody2": "DraftProof 在提交前为你解释这些信号，让你用更强的证据、更清晰的推理和更好的来源支撑来修改。",
  "humanWrittenSignalsLabel": "DraftProof 可以解释的写作信号",
  "humanWrittenSignals": [
    "类似 AI 的可预测性",
    "泛化学术表达",
    "过度平滑的改写",
    "证据和依据支撑薄弱",
    "句子变化不足",
    "缺少作者自身推理"
  ],
  "humanWrittenGuardrail1": "我们不承诺绕过检测器。",
  "humanWrittenGuardrail2": "我们不保证任何 Turnitin 结果。",
  "humanWrittenPunch": "人类亲自写作不一定等于低风险。DraftProof 解释原因。",
  "humanizerEyebrow": "AI 人性化工具的陷阱",
  "humanizerTitle": "“AI 人性化工具”不是解决依据问题的正确方式。",
  "humanizerBody1": "AI“人性化工具”承诺通过打乱词句让文本无法被检测。这样会让文字生硬、模糊你的原意，却没有解决真正的问题：论断仍然缺少清楚的证据或作者自己的推理。",
  "humanizerBody2": "Turnitin 自己的指南说明，这类报告可能误判经过释义或规避工具处理的文本，需要人类判断。DraftProof 不玩这个游戏——我们告诉你哪里要补依据、核实，并用自己的话完成。",
  "humanizerSignalsLabel": "AI 人性化工具实际上在做什么",
  "humanizerSignals": [
    "打乱措辞而不是修复论断",
    "读起来生硬或不一致",
    "悄悄改变你的原意",
    "留下引用缺口",
    "制造规避工具审阅风险",
    "隐藏你本应能解释的写作过程"
  ],
  "humanizerGuardrail1": "DraftProof 不玩规避检测的把戏。",
  "humanizerGuardrail2": "我们诚实地强化你的证据、推理和审阅记录。",
  "humanizerPunch": "拥有论断、来源和措辞，比伪装文本更可靠。",
  "humanizerSourceLabel": "来源：Turnitin AI 写作报告指南",
  "humanizerSourceUrl": "https://guides.turnitin.com/hc/en-us/articles/22774058814093-Using-the-AI-Writing-Report",
  "anchorEyebrow": "内容锚点",
  "anchorTitle": "通过 AI 扫描，不等于已经写出一份有力的提交稿。",
  "anchorBody1": "检测分数可能变低，但读者仍会觉得文章不够好。论断太薄、来源连接不清、缺少作业背景，就没有真实材料可改——只打乱词句，只是在移动同一个弱点。",
  "anchorBody2": "DraftProof 把内容当作根本输入：论断说了什么、哪个来源支撑它、你能解释什么、作业要求什么。最强的修改不是把文字伪装得更顺，而是让这些锚点更清楚地留在文中。",
  "anchorCardsLabel": "DraftProof 审阅的内容锚点",
  "anchorCards": [
    {
      "label": "01",
      "title": "论断内容",
      "body": "先看段落真正想证明什么，而不是只看句子听起来是否自然。"
    },
    {
      "label": "02",
      "title": "来源锚点",
      "body": "明确哪条引用、阅读材料、数据、案例或课堂细节支撑当前论断。"
    },
    {
      "label": "03",
      "title": "作者锚点",
      "body": "保留学生能用自己的推理、观察、方法或课程语境解释的部分。"
    },
    {
      "label": "04",
      "title": "作业锚点",
      "body": "检查答案是否贴合作业题目、评分标准、学科要求和证据类型。"
    }
  ],
  "anchorWorkflowLabel": "修改前的依据流程",
  "anchorWorkflow": [
    "先找到论断，再改变措辞。",
    "把论断连接到来源、例子、方法或作者自己的解释。",
    "最后才修改句子，让更强的内容仍然清楚可见。"
  ],
  "criticalEyebrow": "批判性思维主导",
  "criticalTitle": "主导思考的是你，还是 AI？",
  "criticalBody1": "AI 会很快给出一个自信而工整的答案。教育中的风险不在于用 AI 做研究或头脑风暴，而在于不知不觉把判断交了出去：不加质疑就接受第一个答案，不比较其他可能，也不核对证据。",
  "criticalBody2": "DraftProof 现在会把你的草稿转化为针对你实际所写内容的深化思考提问——你的情境、你的判断、你的推理、你的证据。这些是帮助你深化思考的提问，不是评分，也不是判定。答案由你自己来写。",
  "criticalSignalsLabel": "DraftProof 针对你自己草稿提出的问题",
  // allow-hardcode: static landing-page UI copy (reflective question prompts shown
  // to the reader), mirroring report.criticalThinking.dimensions — never compared
  // against document text, not a scoring/matching oracle.
  "criticalSignals": [
    "这个论点来自哪一份具体的作业、案例或观察？",
    "你可以引用哪个来源或数据来核实它？",
    "是什么推理让你得出这个立场——你是否权衡过其他选择？",
    "你在哪里质疑了第一个答案，而不是原样保留？",
    "这里有哪些内容你能脱离 AI、用自己的话解释？"
  ],
  "criticalGuardrail1": "不是评分，也不是学术不端判定。",
  "criticalGuardrail2": "问题针对你的草稿——答案由你自己来写。",
  "criticalPunch": "真正的防线是主导思考，而不是伪装文字。",
  "studentMisuseEyebrow": "学生常见误解",
  "studentMisuseTitle": "“写烂英文、用坏语法……就能混过去吗？”",
  "studentMisuseBody1": "这是错误方向。学校引入 AI 检查，不是因为流畅英文可疑，而是要看你是否真正理解材料、质疑输出、核查证据，并能解释自己的推理。",
  "studentMisuseBody2": "错别字、坏语法和廉价改词不会制造真实作者身份，只会让作品更弱。更强的提交稿会显示所有权：你的论断、你的来源、你的课程语境、你的例子和你的解释。",
  "studentMisuseSignalsLabel": "学生真正应该修复什么",
  "studentMisuseSignals": [
    "没有真实支撑的薄弱论断",
    "来源与句子不匹配",
    "学生自己解释不清的泛化观点",
    "缺少课程或作业语境",
    "复制例子但没有消化",
    "隐藏作者痕迹的措辞"
  ],
  "studentMisuseGuardrail1": "不要为了显得像人写的而把草稿变差。",
  "studentMisuseGuardrail2": "重新扫描前，先让思考过程清楚可见。",
  "studentMisusePunch": "真正的修复是更清楚的所有权，不是更乱的英文。",
  "sampleEyebrow": "示例报告",
  "reportStrategyCarouselLabel": "示例报告和内容感知修改建议",
  "reportStrategyCarouselTabsLabel": "示例报告和修改建议分区",
  "reportStrategyCarouselNext": "显示下一个报告或建议分区",
  "sampleTitle": "提交前，清楚看到应该修复什么。",
  "sampleBody": "DraftProof 会把扫描结果转化为清晰的审阅计划：哪些地方可能被质疑、哪些论断需要依据、分数为什么变化，以及如何负责任地修改而不丢失自己的原意。",
  "samplePoint1": "解释分数，而不是只看一个百分比猜测",
  "samplePoint2": "优先处理引用、来源、相似度和 AI 风格风险",
  "samplePoint3": "提交前保留一份 PDF 审阅记录",
  "reportValueLabel": "DraftProof 报告带来的价值",
  "reportValueCards": [
    {
      "title": "解释分数",
      "body": "查看 AI 风格写作、依据风险、相似度和作者身份不确定性背后的信号画像。"
    },
    {
      "title": "确定修复优先级",
      "body": "优先处理引用缺口、来源支撑薄弱和影响最大的写作问题。"
    },
    {
      "title": "负责任地修改",
      "body": "把改写建议当作强化依据的示范草稿，然后用你自己的真实细节完成。"
    },
    {
      "title": "保留记录",
      "body": "下载 PDF 报告，记录提交前检查过的内容。"
    }
  ],
  "runOwnScan": "审阅你的内容",
  "helpEyebrow": "DraftProof 如何帮助你",
  "helpTitle": "在正式审阅前准备好你的内容。",
  "helpLead": "DraftProof 会把模糊的提交焦虑转化为清晰的修改计划。它告诉你哪些论断可能被质疑、为什么显得薄弱，以及提交前应该补强什么依据。",
  "whyEyebrow": "为什么学生使用 DraftProof",
  "whyTitle": "Turnitin 改变了学生对原创性、引用和 AI 风格写作的理解。",
  "strategyEyebrow": "内容感知帮助",
  "strategyTitle": "不同作业需要不同修改建议。",
  "strategyBody": "学生草稿、研究段落、政策说明和个人反思不应该用同一种方式修改。DraftProof 会先理解你写了什么，再建议哪里需要补证据、上下文或作者自己的推理。",
  "strategyProofStrong": "DraftProof 不会盲目“人类化”文本。",
  "strategyProofBody": "它帮助学生针对自己真实写出的内容修复引用、来源支撑、相似度和类似 AI 表达风险，并把改写结果视为需要审阅的草稿，而不是直接提交的文本。",
  "engineEyebrow": "工作方式",
  "engineTitle": "四项检查，一份清晰报告。",
  "engineLead": "DraftProof 会在提交前从四个维度分析你的内容。",
  "beliefsEyebrow": "DraftProof 的理念",
  "beliefsTitle": "写作工具应该公平、透明且有用。",
  "positiveBelief": "我们认为用户应该得到清晰、基于证据的反馈，真正帮助他们改进作品。",
  "ctaPill": "提交前，先检查读者可能质疑的地方。",
  "ctaTitle": "DraftProof 帮助你在提交前补强内容依据。",
  "ctaBody": "提交前，检查作品是否原创、引用充分、表达清晰，并由来源或你自己的推理支撑。",
  "ctaButton": "审阅我的内容",
  // allow-hardcode: i18n 展示文案。"5" 对应 WELCOME_CREDITS 默认值；非评分依据。
  "ctaFreeCredits": "新账户注册即送 5 个免费积分，无需信用卡。",
  "ctaSmall": "每 1,000 词 1 个积分 · 包含 PDF 报告 · 不承诺绕过检测",
  "reportPreviewLabel": "DraftProof 示例内容审阅报告预览",
  "reportPreviewTabsLabel": "示例报告部分",
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
  // allow-hardcode: static sample-report UI copy (illustrative reflective questions
  // shown on the landing page), anchored to the fixed sample essay above — never
  // compared against any user document, not a scoring/matching oracle.
  "sampleCriticalQuestions": [
    {
      "quote": "好莱坞已成为历史上最具影响力的文化输出之一。",
      "question": "有哪些具体的影片、数据或来源能证明这一点——你所说的又是在哪个时间段内？"
    },
    {
      "quote": "美国的电影、音乐、时尚和社交媒体潮流在全球范围内被广泛消费。",
      "question": "有哪个来自你自己阅读或观察的例子，能支撑“在全球范围内被消费”，而不只是一个笼统印象？"
    },
    {
      "quote": "美国拥有强大的文化影响力。",
      "question": "是什么推理让你把这种影响力称为“强大”？你是否权衡过任何反例？"
    }
  ],
  "findingsSampleType": "AI 可能性",
  "findingsSampleDescription": "该段落使用了标准过渡词\"除……之外\"以及其他若干常见短语，使文章读起来较为程式化。",
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
  "checks": [
    {
      "title": "引用缺口",
      "body": "识别需要来源但缺少来源的论断。"
    },
    {
      "title": "来源完整性",
      "body": "检查引用来源是否真正支撑所提出的论断。"
    },
    {
      "title": "泛化表达",
      "body": "标记听起来泛化、缺少依据或证据支撑不足的表达。"
    },
    {
      "title": "作者身份信号",
      "body": "呈现仅供审阅的模式，以及草稿中已有的人类锚点。"
    }
  ],
  "helpCards": [
    {
      "title": "找出可能被质疑的地方",
      "body": "在导师或审阅系统看到之前，识别引用缺口、来源支撑薄弱、相似度风险和类似 AI 的写作信号。"
    },
    {
      "title": "理解原因",
      "body": "看清问题来自缺少证据、泛化表达、来源使用不均，还是写作过于整齐，而不是只根据一个分数猜测。"
    },
    {
      "title": "负责任地修改",
      "body": "使用报告和改写来加强论点、补充来源支撑、澄清表达，并保留你自己的原意。"
    },
    {
      "title": "保留审阅记录",
      "body": "下载 PDF 报告，记录提交前检查过什么，以及你做了哪些改进。"
    }
  ],
  "whyCards": [
    {
      "title": "我们正在从合成信息中学习",
      "body": "搜索引擎、聊天机器人和写作助手会先总结知识，然后我们才到达原始来源。写作可能因此与证据脱节。",
      "note": "DraftProof 弥合这个缺口"
    },
    {
      "title": "传统媒体不再是唯一来源",
      "body": "信息现在通过 AI 新闻室、生成式摘要和转述材料流动。表达 polished 不等于已经被证明。",
      "note": "检查来源，也检查论断"
    },
    {
      "title": "只做 AI 检测并不够",
      "body": "分数不是反馈。DraftProof 会问更好的问题：论断是否有支撑，哪里需要修复？",
      "note": "可执行，而不是只给判定"
    }
  ],
  "contentStrategies": [
    {
      "type": "内容草稿",
      "strategy": "让论点更清晰",
      "detail": "保留你的核心观点，并让推理更容易跟随。"
    },
    {
      "type": "带引用的研究写作",
      "strategy": "在不移动来源的情况下改进措辞",
      "detail": "清理薄弱句子，同时保留引用和来源论断的位置。"
    },
    {
      "type": "技术写作",
      "strategy": "保持精确含义不变",
      "detail": "保护术语、步骤、定义和要求，避免过度润色。"
    },
    {
      "type": "法律、政策或医疗文本",
      "strategy": "只做谨慎修改",
      "detail": "在措辞错误可能带来风险的地方，只改真正需要注意的内容。"
    },
    {
      "type": "列表和表格",
      "strategy": "保持结构易读",
      "detail": "改进措辞，同时不破坏行、项目符号、比较或发现。"
    },
    {
      "type": "简短回答",
      "strategy": "添加上下文或要求更多内容",
      "detail": "当文本太短无法判断时说明情况，而不是猜测。"
    },
    {
      "type": "个人反思",
      "strategy": "保留你自己的声音",
      "detail": "保留你的经历和观点，而不是把它变得泛化。"
    },
    {
      "type": "营销文案",
      "strategy": "匹配受众和格式",
      "detail": "改进信息表达，而不是强行变成学术写作风格。"
    }
  ],
  "sampleStats": [
    {
      "label": "风险等级",
      "value": "低风险",
      "tone": "positive"
    },
    {
      "label": "发现总数",
      "value": "50"
    },
    {
      "label": "作者身份评级",
      "value": "良好"
    },
    {
      "label": "原始 AI 风格信号",
      "value": "29%",
      "tone": "positive"
    },
    {
      "label": "写作分数",
      "value": "33%",
      "tone": "accent"
    }
  ],
  "beliefs": [
    "每个像 AI 的句子都不等于不当行为。",
    "每个相似匹配都不等于抄袭。",
    "学生不应该被黑箱分数评判。",
    "把所有内容都改写并不会让写作更诚实。"
  ]
};
