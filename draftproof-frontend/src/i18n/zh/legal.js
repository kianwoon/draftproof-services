export const legal = {
  "lastUpdated": "最后更新",
  "may2026": "2026 年 5 月",
  "privacy": {
    "eyebrow": "隐私政策",
    "title": "我们如何处理你的数据",
    "lead": "DraftProof 会处理学术文档，以提供写作完整性信号和审阅建议。本页说明我们收集什么数据、如何使用，以及你拥有的选择。",
    "stat": "用户可控删除",
    "sections": [
      {
        "title": "我们收集的数据",
        "items": [
          {
            "title": "账户信息",
            "body": "当你使用 Google 或 Microsoft 登录时，我们会从 OAuth 提供方接收你的姓名和邮箱地址。如有头像，我们也会获取头像以个性化体验。"
          },
          {
            "title": "文档",
            "body": "你上传用于扫描的文件，通常是内容草稿、报告或研究文章，会被临时存储，以便我们分析并生成报告。"
          },
          {
            "title": "扫描结果",
            "body": "我们生成的分析报告会被保存，方便你从仪表盘重新查看。"
          },
          {
            "title": "付款信息",
            "body": "我们使用 Stripe 处理付款。我们不会看到或存储你的信用卡号，只保存令牌化确认和积分购买记录。"
          }
        ]
      },
      {
        "title": "我们如何使用数据",
        "bullets": [
          "文档扫描：分析上传文件以生成完整性报告。",
          "报告交付：保存扫描结果，方便你查看和下载。",
          "账户管理：使用 OAuth 提供的邮箱和姓名识别你的账户。",
          "积分计费：Stripe 处理一次性积分购买，我们跟踪你的余额。"
        ]
      },
      {
        "title": "数据存储",
        "paragraphs": [
          "上传文档和生成报告会临时存储在 Cloudflare R2 对象存储中，并启用服务端加密。生成的扫描和改写报告会在 DraftProof 中保留 3 天，之后系统副本会被清理。",
          "应用服务器和数据库托管在 Koyeb 基础设施上，并使用加密连接。所有数据会在我们的托管和存储提供商配置区域内处理和存储。"
        ]
      },
      {
        "title": "第三方服务",
        "table": {
          "headers": [
            "服务",
            "用途",
            "共享数据"
          ],
          "rows": [
            [
              "Google / Microsoft",
              "登录（OAuth 2.0）",
              "姓名、邮箱、头像"
            ],
            [
              "Stripe",
              "付款处理",
              "卡片详情完全由 Stripe 处理"
            ],
            [
              "Cloudflare R2",
              "文件存储",
              "上传文档和报告"
            ]
          ]
        }
      },
      {
        "title": "数据保留与删除",
        "paragraphs": [
          "生成的扫描和改写报告会在 DraftProof 中保留 3 天，之后从系统中清理。启用邮件发送时，报告副本会发送到你的邮箱，方便你自行保存记录。",
          "你拥有控制权。你可以随时从仪表盘删除单个文档和报告。如果你请求删除账户，我们会移除你的个人数据、上传文件和扫描历史。",
          "如需请求删除，请通过下方邮箱联系我们。"
        ]
      },
      {
        "title": "Cookie",
        "paragraphs": [
          "DraftProof 使用一个 httpOnly 会话 Cookie 来维持登录状态。我们不使用跟踪 Cookie、广告像素或第三方分析脚本。"
        ]
      },
      {
        "title": "联系",
        "paragraphs": [
          "对此政策有疑问？请通过 support@draftproof.app 联系我们。"
        ]
      }
    ]
  },
  "security": {
    "eyebrow": "安全",
    "title": "我们如何保护你的作品",
    "lead": "学术文档需要强保护。以下是 DraftProof 在各层保护数据的方式。",
    "stat": "加密存储",
    "sections": [
      {
        "title": "基础设施安全",
        "bullets": [
          "全站 HTTPS：所有到 DraftProof 的连接都使用 TLS 1.2+ 加密。",
          "加密存储：文档和报告存储在 Cloudflare R2，并使用服务端加密（AES-256）。",
          "托管基础设施：应用运行在 Koyeb 基础设施上，具备网络隔离、自动补丁和 DDoS 缓解。"
        ]
      },
      {
        "title": "身份认证",
        "bullets": [
          "OAuth 2.0 登录：支持 Google 和 Microsoft 账户。我们不会看到或存储你的密码。",
          "httpOnly JWT Cookie：会话令牌存储在 httpOnly、Secure、SameSite=Lax Cookie 中。",
          "不存储密码：身份认证委托给 Google 和 Microsoft。"
        ]
      },
      {
        "title": "文档保护",
        "bullets": [
          "静态加密：所有上传文档都在 Cloudflare R2 中使用 AES-256 加密。",
          "传输加密：文件从浏览器到服务器再到存储，全程通过 HTTPS。",
          "用户可控删除：你可以随时从仪表盘删除任何文档或报告。",
          "访问隔离：每个用户只能访问自己的文档和报告。"
        ]
      },
      {
        "title": "付款安全",
        "bullets": [
          "Stripe 处理所有卡片数据：信用卡号不会接触我们的服务器。Stripe 已通过 PCI DSS Level 1 认证。",
          "Webhook 签名验证：Stripe 的付款确认会在处理前进行加密签名验证。",
          "基于积分计费：我们跟踪预付积分余额，不在我们侧进行周期扣费或保存付款方式。"
        ]
      },
      {
        "title": "应用安全",
        "bullets": [
          "输入验证：所有用户输入会在服务器处理前进行验证和清理。",
          "CSRF 防护：SameSite Cookie 策略和 OAuth 流程中的 state 参数可防止跨站请求伪造。",
          "最小攻击面：我们使用聚焦的技术栈（FastAPI + React）和尽量少的依赖。"
        ]
      },
      {
        "title": "负责任披露",
        "paragraphs": [
          "如果你发现 DraftProof 的安全漏洞，我们欢迎负责任披露。请将细节发送至 security@draftproof.app，我们会及时回复。"
        ]
      }
    ]
  }
};
