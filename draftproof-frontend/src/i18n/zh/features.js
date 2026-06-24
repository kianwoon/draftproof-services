// allow-hardcode: user-facing i18n content for /features page UI — not a detector list or logic gate

export const featuresPage = {
  eyebrow: "为什么选择 DraftProof",
  title: "准备。改进。证明。",
  lead: "AI 检测工具在你提交之后才标记你的写作 —— DraftProof 在此之前介入。它帮你准备好草稿，并留存你负责任使用 AI 的证据，而不是替代你所在机构的检测工具。",
  tableLabel: "DraftProof 有何不同",
  competitors: ["DraftProof", "GPTZero", "Turnitin", "Originality.ai", "Winston AI"],
  rows: [
    { label: "段落级别输出", values: ["yes", "yes", "yes", "yes", "no"] },
    { label: "解释内容被标记的原因", values: ["yes", "no", "no", "no", "no"] },
    { label: "集成改写 / 辅导", values: ["yes", "no", "no", "no", "no"] },
    { label: "修改前后对比视图", values: ["yes", "no", "no", "no", "no"] },
    { label: "政策感知评分", values: ["yes", "no", "no", "no", "no"] },
    { label: "提交风险评估", values: ["yes", "no", "no", "no", "no"] },
    { label: "批判性思维评估", values: ["yes", "no", "no", "no", "no"] },
    { label: "如实披露检测局限性", values: ["yes", "no", "no", "no", "no"] },
    { label: "个人可直接使用（无需机构账号）", values: ["yes", "yes", "no", "yes", "yes"] },
  ],
  cardsLabel: "DraftProof 独有功能",
  // allow-hardcode: user-facing feature-card copy for /features — marketing prose, not a detector list or logic gate
  cards: [
    {
      icon: "ti-writing",
      title: "自动生成前后对比改写",
      body: "扫描标记出有风险的段落，改写则为每一处生成一份前后对比 —— 该补充的锚点、来源和细节 —— 让你在提交前用自己的话重写。",
    },
    {
      icon: "ti-brain",
      title: "供你学习的范例",
      body: "改写是一个示范，而不是可直接提交的答案。你看到有依据的写作是什么样子，然后把它变成你自己的。",
    },
    {
      icon: "ti-shield-check",
      title: "政策风险（双模式）",
      body: '同一引擎生成两个评分 —— “允许使用 AI”与“限制使用 AI” —— 匹配你所在机构的实际政策，而非单一通用判定。',
    },
    {
      icon: "ti-clipboard-check",
      title: "提交与思维所有权",
      body: '“你能证明这是你自己的作品吗？” —— 从文本模式、思维所有权和学术依据来评估，并用五个维度告诉你需要更深入思考的方向。',
    },
  ],
  ctaTitle: "以全新视角审视你的写作。",
  ctaBody: "开始你的第一次扫描 — 首次积分免费体验。",
  ctaButton: "开始审阅",
};
