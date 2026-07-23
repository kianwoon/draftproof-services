// allow-hardcode: user-facing i18n content for /features page UI — not a detector list or logic gate

export const featuresPage = {
  eyebrow: "为什么选择 DraftProof",
  title: "准备。改进。证明。",
  lead: "AI 检测工具在你提交之后才标记你的写作 —— DraftProof 在此之前介入，依托一套我们自研自调的引擎：一个经过微调的检测模型、句子级深度扫描，以及会告诉你「为什么」的依据核查 —— 而不只是一个分数。",
  techLabel: "技术内核",
  // allow-hardcode: user-facing tech-pillar copy for /features — marketing prose grounded in
  // the real detector (fine-tune + ESL calibration + deep-scan + claim-graph). Not a scoring
  // oracle. Numbers mirror the vetted public claims (0.8% ESL FPR / 272-essay corpus).
  techPillars: [
    {
      metric: "GPT-5.x · Gemini · Qwen",
      title: "我们自研微调的检测模型",
      body: "并非现成的第三方 API。DraftProof 运行自有模型，在前沿 AI 输出与数千篇真实作文上微调 —— 并随新的前沿模型发布而重新调校，让检测始终跟上学生真正在用的工具。",
    },
    {
      metric: "较低水平 ESL 误报率 0.8%",
      title: "为非母语英语写作者校准",
      body: "检测工具素来容易把流畅的 ESL 写作误判为「AI」。每一次评分变更都要先通过一个 272 篇真实非母语英语作文语料库的检验，只有在误报率保持低位时才会上线 —— 在标准阈值下，较低水平写作者的误报率为 0.8%。",
    },
    {
      metric: "逐句分析",
      title: "深度扫描精确到句子",
      body: "一个独立的深度阅读模型逐句为你的草稿评分，再与模式信号融合 —— 这样任何单一的噪声数值都无法左右判定，你也能清楚看到是哪些句子推高了分数。",
    },
    {
      metric: "蕴含核查",
      title: "论点图谱依据核查",
      body: "DraftProof 梳理你论证中的各项主张，核查每一条是否真正得到所引来源的支撑 —— 揪出那些缺乏依据、泛泛而谈的表述，而这正是写作读起来像 AI 生成的真正原因。",
    },
  ],
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
  tabs: {
    scan: { label: "扫描", desc: "诊断哪些内容被标记、以及为什么 —— 在你提交之前。" },
    rewrite: { label: "改写", desc: "一份前后对比的范例，供你用自己的话重写。" },
  },
  scanCards: [
    {
      icon: "ti-shield-check",
      title: "政策风险（双模式）",
      body: '同一引擎生成两个评分 —— “允许使用 AI”与“限制使用 AI” —— 匹配你所在机构的实际政策，而非单一通用判定。',
    },
    {
      icon: "ti-clipboard-check",
      title: "提交风险评估",
      body: '“你能证明这是你自己的作品吗？” —— 从三个维度评估：文本模式、思维所有权和学术依据。',
    },
    {
      icon: "ti-brain",
      title: "批判性思维控制",
      body: "五个维度 —— 具体情境、学生判断、推理轨迹、证据依据、AI 依赖度 —— 告诉你需要更深入思考的方向。",
    },
    {
      icon: "ti-puzzle",
      title: "在 Word 与 Google 文档中直接使用",
      body: "在你写作的地方直接扫描选中的文字 —— DraftProof 提供 Microsoft Word 与 Google 文档加载项，无需复制粘贴。",
    },
  ],
  rewriteLearnMore: "了解改写的工作方式 →",
  rewriteCards: [
    {
      icon: "ti-writing",
      title: "自动生成前后对比改写",
      body: "扫描标记出有风险的段落，改写则为每一处生成一份前后对比 —— 该补充的锚点、来源和细节 —— 让你在提交前用自己的话重写。",
    },
    {
      icon: "ti-bulb",
      title: "供你学习的范例",
      body: "改写是一个示范，而不是可直接提交的答案。你看到有依据的写作是什么样子，然后把它变成你自己的。",
    },
    {
      icon: "ti-git-compare",
      title: "可直接采用的前后对比",
      body: "高亮的逐段对比清楚展示改了什么、为什么改 —— 让你用自己的话采纳每一处修改。",
    },
  ],
  ctaTitle: "以全新视角审视你的写作。",
  ctaBody: "开始你的第一次扫描 — 首次积分免费体验。",
  ctaButton: "开始审阅",
};
