"""V7 spec §12 validation set — 4-class labeled corpus + category-agreement
measurement for the Authorship Clarity Breakdown.

Classes (construction-labeled, spec §21 seeding — NOT human reviewer labels;
reviewer labeling is the remaining step before the D5 percentage unlock):
- student_owned:        SCoCESLE human ESL essays (local-only license)
- ai_generated_like:    existing AI cases (poc/calibration/authorship_cases)
- ai_assisted_polished: LLM-polished SCoCESLE essays (generated, local-only)
- ai_paraphrased:       LLM-paraphrased SCoCESLE essays (generated, local-only)

Generated variants derive from SCoCESLE text, so they inherit its
no-redistribution license: everything under v12_validation/corpus/ is
gitignored. Only the numbers-only baseline JSON is committable.
"""
