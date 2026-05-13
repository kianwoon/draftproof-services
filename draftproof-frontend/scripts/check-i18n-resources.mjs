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
];

for (const check of checks) {
  if (!check.matches.test(check.value || '')) {
    throw new Error(`${check.label} has unexpected translation: ${check.value}`);
  }
}

console.log(`Validated ${checks.length} i18n resource checks.`);
