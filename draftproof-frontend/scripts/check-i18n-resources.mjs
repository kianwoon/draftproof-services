import { resources } from '../src/i18n/resources.js';

const checks = [
  {
    label: 'English why page',
    value: resources.en.translation.whyPage.title,
    matches: /^We built DraftProof/,
  },
  {
    label: 'Chinese why page',
    value: resources.zh.translation.whyPage.title,
    matches: /^我们建立 DraftProof/,
  },
  {
    label: 'English privacy page',
    value: resources.en.translation.legal.privacy.title,
    matches: /^How we handle your data$/,
  },
  {
    label: 'Chinese privacy page',
    value: resources.zh.translation.legal.privacy.title,
    matches: /^我们如何处理你的数据$/,
  },
  {
    label: 'English security page',
    value: resources.en.translation.legal.security.title,
    matches: /^How we protect your work$/,
  },
  {
    label: 'Chinese security page',
    value: resources.zh.translation.legal.security.title,
    matches: /^我们如何保护你的作品$/,
  },
  {
    label: 'English detector update note',
    value: resources.en.translation.landing.detectorUpdateNote,
    matches: /^Detector freshly fine-tuned on frontier AI writing/,
  },
  {
    label: 'Chinese detector update note',
    value: resources.zh.translation.landing.detectorUpdateNote,
    matches: /^检测器刚完成微调/,
  },
  {
    label: 'English detector update date',
    value: resources.en.translation.landing.detectorUpdateDate,
    matches: /^Updated \w+ 20\d\d$/,
  },
  {
    label: 'Chinese detector update date',
    value: resources.zh.translation.landing.detectorUpdateDate,
    matches: /20\d\d\s*年/,
  },
];

for (const check of checks) {
  if (!check.matches.test(check.value || '')) {
    throw new Error(`${check.label} has unexpected translation: ${check.value}`);
  }
}

console.log(`Validated ${checks.length} i18n resource checks.`);
