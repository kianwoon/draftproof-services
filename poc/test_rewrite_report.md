# DraftProof — Pre-Submission Integrity Report

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Overall Tier** | 🟠 **HIGH** |
| **Total Findings** | 28 |
| Critical | 0 |
| High | 1 |
| Medium | 22 |
| Low | 5 |
| Scan Time | 10.2s |

### Predictability

- **Overall Risk:** `0.4183`
- **Distribution:** {'high': 1, 'medium': 22, 'low': 3}

---

## Rewrite — Comparison

| Metric | Original | Rewritten | Change |
|--------|----------|-----------|--------|
| **Overall Tier** | `HIGH` | `MEDIUM` | `↓ IMPROVED` |
| **Predictability Risk** | `0.4183` | `0.3967` | `+0.0216` |
| **Total Findings** | `28` | `25` | `-3` |
| **HIGH findings** | `1` | `0` | `-1` ← eliminated |
| **MEDIUM findings** | `22` | `22` | `+0` |
| **LOW findings** | `3` | `4` | `+1` |

### Pass Progression

- **Passes:** 3
- **Converged:** No
- **Reason:** Max passes (3) reached

| Pass | Risk | Top-10 | Surprisal |
|------|------|--------|-----------|
| 0 | `0.4183` | `49.6%` | `4.95` |
| 1 | `0.4111` | `47.8%` | `4.96` |
| 2 | `0.3999` | `46.0%` | `5.12` |
| 3 | `0.3967` | `45.8%` | `5.22` |

### Top Rewrite Improvements

| # | Orig | New | Orig Risk | New Risk | Δ Risk | Rewrite |
|---|------|-----|-----------|----------|--------|---------|
| 18 | M | L | `0.4854` | `0.3134` | `-0.1720` ★ | To remain relevant, a professional must ... → A stylist who stops learning after quali... |
| 23 | M | M | `0.5440` | `0.4259` | `-0.1181` ★ | Whether the client sits straighter in th... → Does the client sit a little taller behi... |
| 1 | M | M | `0.4924` | `0.4110` | `-0.0814` | Hairdressing sits at the intersection of... → Hairdressing trades in chemistry, geomet... |
| 21 | M | M | `0.5430` | `0.4789` | `-0.0641` | Many modern professionals are moving tow... → Salons from Utrecht to Ljubljana are swa... |
| 8 | M | M | `0.4865` | `0.4254` | `-0.0611` | Whether performing a complex color corre... → Running a complex colour correction or a... |
| 16 | M | M | `0.4521` | `0.4172` | `-0.0349` | The trade reinvents itself roughly every... → The trade pivots hard roughly once a dec... |
| 19 | M | M | `0.4658` | `0.4412` | `-0.0246` | This involves staying updated on the lat... → That means understanding why a balayage ... |
| 3 | M | M | `0.4856` | `0.4719` | `-0.0137` | Today, a stylist who bleaches a client's... → Today, a stylist who lifts a client's ha... |

### Original Content

```
Hairdressing sits at the intersection of chemistry, geometry, and gut instinct — a trade where a 2 mm trimming error can turn a client into an ex-client. From the curled wigs of Versailles courtiers to the fade cuts of 1990s Brooklyn barbershops, hair has carried meaning far beyond aesthetics. Today, a stylist who bleaches a client's hair to platinum on Monday must still ensure it doesn't snap off by Thursday — equal parts chemist and sculptor.

The craft rests on a paradox. Hold a section at 90 degrees with too much tension and the graduation disappears; hold it at 45 with too little and the client gets a shelf instead of layers. I once watched a junior stylist learn this the hard way on a Friday afternoon — the client was her landlord. A bleach bath left on two minutes too long at pH 11 will lift pigment and melt the disulfide bonds in the same stroke. Whether performing a complex color correction or a chemical straightening treatment, the professional must possess a deep understanding of how different products interact with the hair's keratin structure. Miss that balance and the hair goes gummy — technically achieved but structurally wrecked.

Strip away the foils and blow-dryers and you are left with two strangers in a mirror, one of them holding scissors. A client will say just a trim and mean make me look like I did at 22. Perhaps the hardest skill in the trade is hearing the gap between those two sentences. Fine, colour-treated hair won't hold a bleached pixie cut no matter how many reference photos arrive on Instagram — and saying so without losing the booking is its own art. Few other professionals stand close enough to notice a client's hands shaking after a breakup — and the chair doesn't allow small talk to stay small for long. A stylist who remembers that a client's mother preferred a side part — and why — keeps that chair booked for years.

The trade reinvents itself roughly every decade. A bob that takes off on TikTok on Monday can fill appointment books by Friday. To remain relevant, a professional must commit to lifelong learning. This involves staying updated on the latest tools, such as ergonomic shears or high-tech heat styling equipment, as well as mastering new techniques like balayage, foliage, or precision barbering. Sustainability has moved from marketing copy to licensing requirements in parts of the EU. Many modern professionals are moving toward green salon practices, utilizing biodegradable products and reducing water waste to minimize their environmental footprint.

The real metric? Whether the client sits straighter in the car on the way home. A colour correction after a botched box dye job has been known to change which meetings someone volunteers to speak at. It is one of the few trades where steel meets keratin meets psychology — and has done since barbers also pulled teeth. A fringe trimmed 5 mm too short or a copper toner left on 30 seconds too long can alter how a stranger reads your face
```

### Rewritten Content

```
Hairdressing trades in chemistry, geometry, and nerve — misjudge by two millimetres and the client walks. From the curled wigs of Versailles courtiers to the fade cuts of 1990s Brooklyn barbershops, hair has carried meaning far beyond aesthetics. Today, a stylist who lifts a client's hair to platinum on Monday still has to guarantee it flexes on Thursday — half bench chemist, half sculptor.

The craft rests on a paradox. Hold a section at 90 degrees with too much tension and the graduation disappears; hold it at 45 with too little and the client gets a shelf instead of layers. I once watched a junior stylist learn this the hard way on a Friday afternoon — the client was her landlord. A bleach bath left on two minutes too long at pH 11 will lift pigment and melt the disulfide bonds in the same stroke. Running a complex colour correction or a chemical relaxer without knowing how the product attacks keratin is not bold — it is negligent. Miss that balance and the hair goes gummy — technically achieved but structurally wrecked.

Peel back the foils and the chair becomes what it always was: two people who barely know each other, one of them holding scissors. A client will say just a trim and mean make me look like I did at 22. Perhaps the hardest skill in the trade is hearing the gap between those two sentences. Fine, colour-treated hair won't hold a bleached pixie cut no matter how many reference photos arrive on Instagram — and saying so without losing the booking is its own art. Few other professionals stand close enough to notice a client's hands shaking after a breakup — and the chair doesn't allow small talk to stay small for long. A stylist who remembers that a client's mother preferred a side part — and why — keeps that chair booked for years.

The trade pivots hard roughly once a decade. A bob that takes off on TikTok on Monday can fill appointment books by Friday. A stylist who stops learning after qualifying might as well hand back the licence. That means understanding why a balayage pattern behaves differently on coarse versus fine hair, not just buying the newest ergonomic shears on the market. Sustainability has moved from marketing copy to licensing requirements in parts of the EU. Salons from Utrecht to Ljubljana are swapping ammonia-based colour for plant-derived lines, and at least three EU member states now tie water-recycling to licensing.

The real metric? Does the client sit a little taller behind the wheel on the drive back? A colour correction after a botched box dye job has been known to change which meetings someone volunteers to speak at. Steel on keratin on human psychology — few trades run all three at once, and barbers have done it since they also pulled teeth. A fringe trimmed 5 mm too short or a copper toner left on 30 seconds too long can alter how a stranger reads your face
```

---

## Detection Findings (pre-rewrite)

### High (1)

**1. [predictability] high_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.5523 predictability risk (top-10 ratio: 70.4%)
- **Evidence:** > Strip away the foils and blow-dryers and you are left with two strangers in a mirror, one of them holding scissors.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.5523`
  - `avg_probability`: `0.179302`
  - `avg_surprisal`: `3.8528`
  - `top10_ratio`: `0.7037`
  - `top50_ratio`: `0.7778`

### Medium (22)

**1. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.4924 predictability risk (top-10 ratio: 61.8%)
- **Evidence:** > Hairdressing sits at the intersection of chemistry, geometry, and gut instinct — a trade where a 2 mm trimming error can
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.4924`
  - `avg_probability`: `0.174549`
  - `avg_surprisal`: `4.2705`
  - `top10_ratio`: `0.6176`
  - `top50_ratio`: `0.7059`

**2. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.4449 predictability risk (top-10 ratio: 53.3%)
- **Evidence:** > From the curled wigs of Versailles courtiers to the fade cuts of 1990s Brooklyn barbershops, hair has carried meaning fa
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.4449`
  - `avg_probability`: `0.293732`
  - `avg_surprisal`: `4.2253`
  - `top10_ratio`: `0.5333`
  - `top50_ratio`: `0.6667`

**3. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.4856 predictability risk (top-10 ratio: 57.6%)
- **Evidence:** > Today, a stylist who bleaches a client's hair to platinum on Monday must still ensure it doesn't snap off by Thursday — 
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.4856`
  - `avg_probability`: `0.188138`
  - `avg_surprisal`: `4.389`
  - `top10_ratio`: `0.5758`
  - `top50_ratio`: `0.7576`

**4. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.3776 predictability risk (top-10 ratio: 50.0%)
- **Evidence:** > The craft rests on a paradox.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.3776`
  - `avg_probability`: `0.14169`
  - `avg_surprisal`: `6.2373`
  - `top10_ratio`: `0.5`
  - `top50_ratio`: `0.5`

**5. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.4685 predictability risk (top-10 ratio: 54.8%)
- **Evidence:** > Hold a section at 90 degrees with too much tension and the graduation disappears; hold it at 45 with too little and the 
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.4685`
  - `avg_probability`: `0.177012`
  - `avg_surprisal`: `4.515`
  - `top10_ratio`: `0.5484`
  - `top50_ratio`: `0.7419`

**6. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.4352 predictability risk (top-10 ratio: 45.5%)
- **Evidence:** > I once watched a junior stylist learn this the hard way on a Friday afternoon — the client was her landlord.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.4352`
  - `avg_probability`: `0.200447`
  - `avg_surprisal`: `4.3402`
  - `top10_ratio`: `0.4545`
  - `top50_ratio`: `0.7727`

**7. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.3663 predictability risk (top-10 ratio: 42.3%)
- **Evidence:** > A bleach bath left on two minutes too long at pH 11 will lift pigment and melt the disulfide bonds in the same stroke.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.3663`
  - `avg_probability`: `0.151209`
  - `avg_surprisal`: `5.3127`
  - `top10_ratio`: `0.4231`
  - `top50_ratio`: `0.5769`

**8. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.4865 predictability risk (top-10 ratio: 59.4%)
- **Evidence:** > Whether performing a complex color correction or a chemical straightening treatment, the professional must possess a dee
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.4865`
  - `avg_probability`: `0.167867`
  - `avg_surprisal`: `4.0444`
  - `top10_ratio`: `0.5938`
  - `top50_ratio`: `0.7188`

**9. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.3631 predictability risk (top-10 ratio: 41.2%)
- **Evidence:** > A client will say just a trim and mean make me look like I did at 22.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.3631`
  - `avg_probability`: `0.040561`
  - `avg_surprisal`: `5.5027`
  - `top10_ratio`: `0.4118`
  - `top50_ratio`: `0.5882`

**10. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.4755 predictability risk (top-10 ratio: 60.0%)
- **Evidence:** > Perhaps the hardest skill in the trade is hearing the gap between those two sentences.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.4755`
  - `avg_probability`: `0.154157`
  - `avg_surprisal`: `4.1504`
  - `top10_ratio`: `0.6`
  - `top50_ratio`: `0.6667`

**11. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.4464 predictability risk (top-10 ratio: 52.8%)
- **Evidence:** > Fine, colour-treated hair won't hold a bleached pixie cut no matter how many reference photos arrive on Instagram — and 
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.4464`
  - `avg_probability`: `0.170906`
  - `avg_surprisal`: `4.6686`
  - `top10_ratio`: `0.5278`
  - `top50_ratio`: `0.6944`

**12. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.3968 predictability risk (top-10 ratio: 43.3%)
- **Evidence:** > Few other professionals stand close enough to notice a client's hands shaking after a breakup — and the chair doesn't al
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.3968`
  - `avg_probability`: `0.151046`
  - `avg_surprisal`: `4.6878`
  - `top10_ratio`: `0.4333`
  - `top50_ratio`: `0.6667`

**13. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.3864 predictability risk (top-10 ratio: 41.7%)
- **Evidence:** > A stylist who remembers that a client's mother preferred a side part — and why — keeps that chair booked for years.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.3864`
  - `avg_probability`: `0.068001`
  - `avg_surprisal`: `5.2104`
  - `top10_ratio`: `0.4167`
  - `top50_ratio`: `0.6667`

**14. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.4521 predictability risk (top-10 ratio: 55.6%)
- **Evidence:** > The trade reinvents itself roughly every decade.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.4521`
  - `avg_probability`: `0.199785`
  - `avg_surprisal`: `4.6405`
  - `top10_ratio`: `0.5556`
  - `top50_ratio`: `0.6667`

**15. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.3516 predictability risk (top-10 ratio: 37.5%)
- **Evidence:** > A bob that takes off on TikTok on Monday can fill appointment books by Friday.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.3516`
  - `avg_probability`: `0.029177`
  - `avg_surprisal`: `6.5131`
  - `top10_ratio`: `0.375`
  - `top50_ratio`: `0.625`

**16. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.4854 predictability risk (top-10 ratio: 63.6%)
- **Evidence:** > To remain relevant, a professional must commit to lifelong learning.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.4854`
  - `avg_probability`: `0.13298`
  - `avg_surprisal`: `4.0117`
  - `top10_ratio`: `0.6364`
  - `top50_ratio`: `0.6364`

**17. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.4658 predictability risk (top-10 ratio: 58.5%)
- **Evidence:** > This involves staying updated on the latest tools, such as ergonomic shears or high-tech heat styling equipment, as well
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.4658`
  - `avg_probability`: `0.225186`
  - `avg_surprisal`: `4.2952`
  - `top10_ratio`: `0.5854`
  - `top50_ratio`: `0.6585`

**18. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.3887 predictability risk (top-10 ratio: 43.8%)
- **Evidence:** > Sustainability has moved from marketing copy to licensing requirements in parts of the EU.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.3887`
  - `avg_probability`: `0.240057`
  - `avg_surprisal`: `4.6161`
  - `top10_ratio`: `0.4375`
  - `top50_ratio`: `0.625`

**19. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.5430 predictability risk (top-10 ratio: 72.0%)
- **Evidence:** > Many modern professionals are moving toward green salon practices, utilizing biodegradable products and reducing water w
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.543`
  - `avg_probability`: `0.205155`
  - `avg_surprisal`: `4.1234`
  - `top10_ratio`: `0.72`
  - `top50_ratio`: `0.72`

**20. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.5440 predictability risk (top-10 ratio: 69.2%)
- **Evidence:** > Whether the client sits straighter in the car on the way home.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.544`
  - `avg_probability`: `0.185716`
  - `avg_surprisal`: `3.9845`
  - `top10_ratio`: `0.6923`
  - `top50_ratio`: `0.7692`

**21. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.3822 predictability risk (top-10 ratio: 45.8%)
- **Evidence:** > It is one of the few trades where steel meets keratin meets psychology — and has done since barbers also pulled teeth.
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.3822`
  - `avg_probability`: `0.096017`
  - `avg_surprisal`: `5.6408`
  - `top10_ratio`: `0.4583`
  - `top50_ratio`: `0.5833`

**22. [predictability] medium_predictability**

- **Scanner:** `predictability`
- **Detail:** Sentence scored 0.3689 predictability risk (top-10 ratio: 40.0%)
- **Evidence:** > A fringe trimmed 5 mm too short or a copper toner left on 30 seconds too long can alter how a stranger reads your face
- **Recommendation:** Add specific evidence, cited claims, or original phrasing.
  - `score`: `0.3689`
  - `avg_probability`: `0.078008`
  - `avg_surprisal`: `5.9122`
  - `top10_ratio`: `0.4`
  - `top50_ratio`: `0.64`

### Low (5)

**1. [predictability] style_shift**

- **Scanner:** `predictability`
- **Detail:** Predictability less_predictable (Δ0.2088)
- **Evidence:** > Miss that balance and the hair goes gummy — technically achieved but structurall
- **Recommendation:** Review for consistency in writing voice.
  - `from`: `Whether performing a complex color correction or a chemical straightening treatm`
  - `to`: `Miss that balance and the hair goes gummy — technically achieved but structurall`
  - `magnitude`: `0.2088`
  - `direction`: `less_predictable`

**2. [predictability] style_shift**

- **Scanner:** `predictability`
- **Detail:** Predictability more_predictable (Δ0.2746)
- **Evidence:** > Strip away the foils and blow-dryers and you are left with two strangers in a mi
- **Recommendation:** Review for consistency in writing voice.
  - `from`: `Miss that balance and the hair goes gummy — technically achieved but structurall`
  - `to`: `Strip away the foils and blow-dryers and you are left with two strangers in a mi`
  - `magnitude`: `0.2746`
  - `direction`: `more_predictable`

**3. [predictability] style_shift**

- **Scanner:** `predictability`
- **Detail:** Predictability less_predictable (Δ0.4369)
- **Evidence:** > The real metric?
- **Recommendation:** Review for consistency in writing voice.
  - `from`: `Many modern professionals are moving toward green salon practices, utilizing bio`
  - `to`: `The real metric?`
  - `magnitude`: `0.4369`
  - `direction`: `less_predictable`

**4. [predictability] style_shift**

- **Scanner:** `predictability`
- **Detail:** Predictability more_predictable (Δ0.4379)
- **Evidence:** > Whether the client sits straighter in the car on the way home.
- **Recommendation:** Review for consistency in writing voice.
  - `from`: `The real metric?`
  - `to`: `Whether the client sits straighter in the car on the way home.`
  - `magnitude`: `0.4379`
  - `direction`: `more_predictable`

**5. [predictability] style_shift**

- **Scanner:** `predictability`
- **Detail:** Predictability less_predictable (Δ0.2113)
- **Evidence:** > A colour correction after a botched box dye job has been known to change which m
- **Recommendation:** Review for consistency in writing voice.
  - `from`: `Whether the client sits straighter in the car on the way home.`
  - `to`: `A colour correction after a botched box dye job has been known to change which m`
  - `magnitude`: `0.2113`
  - `direction`: `less_predictable`

---

<details>
<summary>📊 Predictability — Per-Sentence Breakdown</summary>

| # | Risk | Score | Surprisal | Top-10 | Sentence |
|---|------|-------|-----------|--------|----------|
| 1 | MEDIUM | `0.4924` | `4.27` | `61.8%` | Hairdressing sits at the intersection of chemistry... |
| 2 | MEDIUM | `0.4449` | `4.23` | `53.3%` | From the curled wigs of Versailles courtiers to th... |
| 3 | MEDIUM | `0.4856` | `4.39` | `57.6%` | Today, a stylist who bleaches a client's hair to p... |
| 4 | MEDIUM | `0.3776` | `6.24` | `50.0%` | The craft rests on a paradox.... |
| 5 | MEDIUM | `0.4685` | `4.51` | `54.8%` | Hold a section at 90 degrees with too much tension... |
| 6 | MEDIUM | `0.4352` | `4.34` | `45.5%` | I once watched a junior stylist learn this the har... |
| 7 | MEDIUM | `0.3663` | `5.31` | `42.3%` | A bleach bath left on two minutes too long at pH 1... |
| 8 | MEDIUM | `0.4865` | `4.04` | `59.4%` | Whether performing a complex color correction or a... |
| 9 | LOW | `0.2777` | `6.22` | `31.2%` | Miss that balance and the hair goes gummy — techni... |
| 10 | HIGH | `0.5523` | `3.85` | `70.4%` | Strip away the foils and blow-dryers and you are l... |
| 11 | MEDIUM | `0.3631` | `5.50` | `41.2%` | A client will say just a trim and mean make me loo... |
| 12 | MEDIUM | `0.4755` | `4.15` | `60.0%` | Perhaps the hardest skill in the trade is hearing ... |
| 13 | MEDIUM | `0.4464` | `4.67` | `52.8%` | Fine, colour-treated hair won't hold a bleached pi... |
| 14 | MEDIUM | `0.3968` | `4.69` | `43.3%` | Few other professionals stand close enough to noti... |
| 15 | MEDIUM | `0.3864` | `5.21` | `41.7%` | A stylist who remembers that a client's mother pre... |
| 16 | MEDIUM | `0.4521` | `4.64` | `55.6%` | The trade reinvents itself roughly every decade.... |
| 17 | MEDIUM | `0.3516` | `6.51` | `37.5%` | A bob that takes off on TikTok on Monday can fill ... |
| 18 | MEDIUM | `0.4854` | `4.01` | `63.6%` | To remain relevant, a professional must commit to ... |
| 19 | MEDIUM | `0.4658` | `4.30` | `58.5%` | This involves staying updated on the latest tools,... |
| 20 | MEDIUM | `0.3887` | `4.62` | `43.8%` | Sustainability has moved from marketing copy to li... |
| 21 | MEDIUM | `0.5430` | `4.12` | `72.0%` | Many modern professionals are moving toward green ... |
| 22 | LOW | `0.1061` | `7.80` | `0.0%` | The real metric?... |
| 23 | MEDIUM | `0.5440` | `3.98` | `69.2%` | Whether the client sits straighter in the car on t... |
| 24 | LOW | `0.3327` | `5.59` | `38.1%` | A colour correction after a botched box dye job ha... |
| 25 | MEDIUM | `0.3822` | `5.64` | `45.8%` | It is one of the few trades where steel meets kera... |
| 26 | MEDIUM | `0.3689` | `5.91` | `40.0%` | A fringe trimmed 5 mm too short or a copper toner ... |

</details>

<details>
<summary>🔀 Style Shifts</summary>

| # | Location | Shift Score |
|---|----------|-------------|
| 1 | Whether performing a complex c… → Miss that balance and the hair… (less_predictable) | `0.2088` |
| 2 | Miss that balance and the hair… → Strip away the foils and blow-… (more_predictable) | `0.2746` |
| 3 | Many modern professionals are … → The real metric?… (less_predictable) | `0.4369` |
| 4 | The real metric?… → Whether the client sits straig… (more_predictable) | `0.4379` |
| 5 | Whether the client sits straig… → A colour correction after a bo… (less_predictable) | `0.2113` |

</details>

---

> **Note:** This is a pre-submission integrity check, not a plagiarism or AI-authorship verdict. Signals should be reviewed in context.