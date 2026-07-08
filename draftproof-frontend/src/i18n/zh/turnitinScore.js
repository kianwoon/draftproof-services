// allow-hardcode: human-crafted SEO landing-page copy (marketing prose), not a scoring/matching oracle
export const turnitinScore = {
  "eyebrow": "Turnitin AI 分数",
  "title": "你的 Turnitin AI 分数到底意味着什么？",
  "lead": "Turnitin AI 报告顶部的百分比，是学术界最被误解的数字之一。它不是抄袭分数，不是相似度分数，也不是学术不端的证据。下面诚实地告诉你该怎么读它。",
  "ctaPrimaryLabel": "检查你自己的写作",
  "ctaSecondaryLabel": "查看内容检查工具",
  "heroStat": {
    "label": "适合用于",
    "value": "理解被标记的原因",
    "detail": "在你慌张或申辩之前"
  },
  "intro": [
    {
      "title": "这个百分比是什么",
      "body": "它是 Turnitin 对你符合条件的文本中，有多少与其模型认定的 AI 写作模式相匹配的估计值。是一个估计——而产生它的模型，可能出错。"
    },
    {
      "title": "它不是什么",
      "body": "不是抄袭或相似度分数，不是衡量你写得好不好，单凭它本身也不是学术不端的证据——Turnitin 自己也说，分数需要人工判断。"
    }
  ],
  "sections": [
    {
      "eyebrow": "读懂数字",
      "title": "这些百分比究竟代表什么。",
      "type": "cards",
      "items": [
        {
          "title": "20% 的下限",
          "body": "Turnitin 会把 1–19% 的分数隐藏在星号后面，因为这个区间的检测置信度低、误报更多。空白或星号并不代表你“干净”——只是信号太弱，不足以报告。"
        },
        {
          "title": "更高的百分比",
          "body": "数字越高，代表越多文本匹配到类似 AI 的模式——而不是某个具体段落一定是 AI 写的。模型标记的是风格，不是作者身份。"
        },
        {
          "title": "误报真实存在",
          "body": "朴素、结构工整或非母语的英文写作，都可能被读成“类似 AI”。真正由人写的内容也会被标记。分数是提示你再看一眼，而不是定论。"
        }
      ]
    },
    {
      "eyebrow": "为什么会这样",
      "title": "为什么人类写作会被标记。",
      "type": "cards",
      "items": [
        {
          "title": "笼统、无依据的主张",
          "body": "那些虽然正确、但放在任何论文里都成立的句子——没有具体细节、没有来源——在统计上显得可预测，而这正是检测器会反应的地方。"
        },
        {
          "title": "过度打磨的表达",
          "body": "大量编辑、语法工具和模板化结构，会抹平人类写作天然的起伏与不平整。"
        },
        {
          "title": "缺少证据",
          "body": "没有引用、缺乏具体依据的主张，是首要的风险信号——比任何单个用词都更关键。"
        }
      ]
    },
    {
      "eyebrow": "该怎么做",
      "title": "被标记之后，真正有用的做法。",
      "type": "steps",
      "items": [
        {
          "title": "不要只是换词",
          "body": "同义词替换保留了同样笼统、无依据的主张——也就是检测器真正会反应的东西。它很少能改变分数，还常常让写作变得更糟。"
        },
        {
          "title": "补上只有你才有的具体内容",
          "body": "用真实的来源、例子和你自己的推理为主张提供依据。具体、有依据的写作既更有力，也更不像 AI。"
        },
        {
          "title": "提交前先检查",
          "body": "用 DraftProof 跑一遍你的草稿，逐段看清哪些段落在拉高风险、究竟缺了什么。"
        }
      ]
    }
  ],
  "linksEyebrow": "继续阅读",
  "linksTitle": "相关指南",
  "links": [
    {
      "label": "检测器为何会标记人类写作",
      "to": "/why",
      "body": "误报背后的机制。"
    },
    {
      "label": "Turnitin 与其他 AI 检测工具对比",
      "to": "/turnitin-vs-ai-detectors",
      "body": "Turnitin、GPTZero 与 Originality.ai 究竟有何不同。"
    },
    {
      "label": "如何降低 AI 检测（诚实版）",
      "to": "/reduce-ai-detection",
      "body": "为什么“骗过检测器”是错误的目标。"
    },
    {
      "label": "学术诚信与 AI",
      "to": "/academic-integrity-ai",
      "body": "哪些 AI 用法没问题，哪些不行。"
    }
  ],
  "ctaTitle": "看清是什么在拉高你的分数。",
  "ctaBody": "DraftProof 会告诉你哪些段落看起来有风险、缺了什么——让你修正写作本身，而不是去操纵那个数字。"
};
