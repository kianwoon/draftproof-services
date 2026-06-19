// allow-hardcode: user-facing i18n content for /features page UI — not a detector list or logic gate

export const featuresPage = {
  eyebrow: "为什么选择 DraftProof",
  title: "检测。理解。提升。",
  lead: "其他检测工具只告诉你不合格。DraftProof 告诉你如何通过 — 通过教你更好地写作。",
  tableLabel: "功能对比",
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
  cards: [
    {
      icon: "ti-writing",
      title: "基于依据的改写辅导",
      body: "展示你段落的改写示例，包含具体锚点、来源和细节 — 让你看到有依据的写作是什么样子。",
    },
    {
      icon: "ti-shield-check",
      title: "政策风险（双模式）",
      body: '同一引擎生成两个评分 — “允许使用 AI”与“限制使用 AI” — 匹配你所在机构的实际政策，而非单一通用判定。',
    },
    {
      icon: "ti-clipboard-check",
      title: "提交风险评估",
      body: '“你能证明这是你自己的作品吗？” — 从三个维度评估：文本模式、思维所有权和学术依据。',
    },
    {
      icon: "ti-brain",
      title: "批判性思维控制",
      body: "五个维度 — 具体情境、学生判断、推理轨迹、证据依据、AI 依赖度 — 告诉你需要更深入思考的方向。",
    },
  ],
  ctaTitle: "以全新视角审视你的写作。",
  ctaBody: "开始你的第一次扫描 — 首次积分免费体验。",
  ctaButton: "开始审阅",
};
