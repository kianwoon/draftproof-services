// allow-hardcode: human-crafted SEO landing-page copy + adaptable declaration templates (user-fill placeholders), not a scoring/matching oracle
export const aiDeclaration = {
  "eyebrow": "AI 使用声明",
  "title": "如何声明你的 AI 使用（附范例）。",
  "lead": "越来越多课程要求你说明用过的 AI 工具。一份清楚的声明能保护你，只需两分钟。下面是可直接复制、再做调整的模板——参照常见院校的致谢格式。",
  "ctaPrimaryLabel": "检查你的草稿",
  "ctaSecondaryLabel": "查看内容检查工具",
  "heroStat": {
    "label": "适合用于",
    "value": "披露 AI 的使用",
    "detail": "简短、具体、诚实的致谢"
  },
  // allow-hardcode: human-crafted declaration-builder UI strings + user-fill template, not a scoring/matching oracle
  "generator": {
    "eyebrow": "声明生成器",
    "title": "生成你的声明。",
    "subtitle": "填好空格并复制你的声明。它只留在你的设备上——不会被发送或存储。",
    "toolLabel": "你使用的工具",
    "toolPlaceholder": "例如 ChatGPT",
    "purposeLabel": "你用它来做什么",
    "purposePlaceholder": "例如 头脑风暴并搭建提纲",
    "dateLabel": "日期（可选）",
    "datePlaceholder": "例如 2026 年 3 月 12 日",
    "outputLabel": "你的声明",
    "copy": "复制声明",
    "copied": "已复制！",
    "template": "在准备这份作业的过程中，我使用了 {{tool}} 以 {{purpose}}{{dateClause}}。在使用该工具后，我已按需审阅并编辑了内容，并对本次提交的内容承担全部责任。",
    "dateClause": "（于 {{date}}）",
    "emptyTool": "[工具]",
    "emptyPurpose": "[用途]"
  },
  "intro": [
    {
      "title": "为什么要声明？",
      "body": "披露能把“你用了 AI 吗？”从一项指控，变成根本不成问题的事。它体现诚信，符合越来越多硬性要求的规定，并在你的作品被质疑时保护你。"
    },
    {
      "title": "该写些什么",
      "body": "三件事：你用了哪个工具、用来做什么、以及（通常还有）什么时候用的。保持简短具体。务必声明你已审阅其输出，并对最终作品负责。"
    }
  ],
  "sections": [
    {
      "eyebrow": "可复制模板",
      "title": "可直接调整的声明范例。",
      "lead": "替换方括号内的部分。你课程要求的措辞，始终优先于通用模板。",
      "type": "templates",
      "items": [
        {
          "title": "通用（Elsevier 风格）",
          "body": "在准备这份作业的过程中，我使用了 [工具] 以 [用途]。在使用该工具后，我已按需审阅并编辑了内容，并对本次提交的内容承担全部责任。",
          "note": "最被广泛接受的格式——调整方括号部分即可。"
        },
        {
          "title": "ChatGPT（构思辅助）",
          "body": "我声明在 [日期] 使用了 ChatGPT（OpenAI，https://chat.openai.com）来进行头脑风暴并搭建本作业的提纲。所有最终文本、分析与结论均为我本人完成。",
          "note": "适用于 AI 帮你做规划、而非代笔的情况。"
        },
        {
          "title": "Grammarly（仅校对）",
          "body": "我声明使用了 Grammarly 来检查本作业的语法、拼写和清晰度。该工具未生成任何内容或论点。",
          "note": "适用于仅做校对的情况。"
        },
        {
          "title": "Microsoft Copilot",
          "body": "我声明使用了 Microsoft Copilot 来归纳我自己的笔记并校对最终草稿。我已审阅全部输出，并对所提交的作品承担全部责任。",
          "note": "参照常见院校致谢范例。"
        },
        {
          "title": "研究 / 归纳来源",
          "body": "我在 [日期] 使用了 [工具] 来归纳我自己挑选的来源。我已对照原文核实每一条归纳，所有引用均为我本人完成。",
          "note": "强调你已核实——AI 归纳可能曲解来源。"
        },
        {
          "title": "未使用 AI",
          "body": "本作业的准备过程中未使用任何生成式 AI 工具。",
          "note": "一份“干净”的声明，也是一种声明。"
        }
      ]
    },
    {
      "eyebrow": "把它写对",
      "title": "常见的声明误区。",
      "type": "cards",
      "items": [
        {
          "title": "先看你的题目要求",
          "body": "有些评估要求特定格式或单独的表格。你课程的措辞，始终优先于通用模板。"
        },
        {
          "title": "要具体，不要含糊",
          "body": "“我用了一点 AI”帮不了任何人。说清工具和用途。具体本身就读作诚实。"
        },
        {
          "title": "别用过度声明来掩盖薄弱内容",
          "body": "声明说明的是你的过程，并不能为无依据的主张或伪造的引用开脱。写作本身也要修好。"
        }
      ]
    }
  ],
  "linksEyebrow": "继续阅读",
  "linksTitle": "相关指南",
  "links": [
    {
      "label": "学术诚信与 AI",
      "to": "/academic-integrity-ai",
      "body": "哪些 AI 用法没问题，哪些不行。"
    },
    {
      "label": "你的 Turnitin AI 分数意味着什么",
      "to": "/turnitin-ai-score",
      "body": "诚实地读懂这个百分比。"
    },
    {
      "label": "检查你的写作",
      "to": "/content-checker",
      "body": "在提交前找出依据薄弱之处。"
    }
  ],
  "ctaTitle": "声明完了？接着把它写扎实。",
  "ctaBody": "声明覆盖的是过程，DraftProof 检查的是实质——引用、依据和类似 AI 的段落——让作品本身站得住脚。"
};
