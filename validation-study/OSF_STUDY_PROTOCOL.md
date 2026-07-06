# Human Rater Validation Study — Study Protocol and Analysis Plan

**Study title.** Human rater validation of an LLM rubric judge for AI tutoring dialogues (rubric items R5 and R2), a component of the IJAIED 2026 manuscript "Dovetailing Case Based Reasoning and Large Language Models to Compare Teaching Strategies" by Dietmar Janetzko and Horacio González-Vélez.

**Deposit context.** This document is a **timestamped study protocol and analysis plan deposit**. It is not a pre-registration in the technical sense: the deposit is being made after data collection, at the time of manuscript resubmission. The purpose of the deposit is to create a public, timestamped, and verifiable record of the sampling frame, exclusion rules, statistical analysis, and decision structure that the analysis code applied. The corresponding code is at `https://github.com/CCC-NCI/mnemonic-cbr` under `mnemonic-cbr/code/analysis/human_validation.py`.

**Deposit date.** 2026-07-04.

---

## 1. Study purpose

The study anchors the LLM rubric judge used in the parent manuscript against human raters on two rubric items:

- **R5 (terminal reasoning state, primary outcome).** The student's reasoning at the closing turn of the five turn dialogue.
- **R2 (cognitive demand, secondary anchored item).** Whether the teacher demands reasoning or supplies answers.

The remaining rubric items (R1, R3, R4, R6) are reported as LLM judge scores and were not part of the human validation study.

---

## 2. Sampling frame

**Source dataset.** The 1{,}250 dialogues produced by the framework's five turn Phase B run (25 EEDI cases × 5 architectures × 5 personas × 2 student LLM legs), scored by the primary LLM judge (Claude Sonnet). Manifest: `mnemonic-cbr/results/phaseB_smoke_5turn/manifest.sqlite`.

**Sampling design.** Stratified subset of 150 dialogues drawn from the 1{,}250 dialogue dataset. Strata: architecture (5 levels) × LLM judge R5 band (low: R5 ≤ 2, mid: 2 < R5 < 3.5, high: R5 ≥ 3.5), giving 15 strata at approximately 10 dialogues each. Persona and student leg balanced within strata.

**Rationale for stratification on LLM judge score.** R5 is floor compressed in the underlying data (grand mean ≈ 2.67). Sampling across score bands guarantees variance in the rated material so that low agreement estimates reflect disagreement rather than range compression. The cost of this choice is stated in advance: an ICC computed on a range stratified sample is not a population estimate and will run higher than one on a random sample.

**Seed and hash.** Sampling seed = 42. The frozen sampling frame manifest is `validation-study/sampling_frame/sampling_frame_manifest.json`, with SHA-256:

```
cb679ed22e5f81784a79c049098947b93f17236b6373775dfe6d14b2a3996904
```

**Sentinels.** Two additional sentinel dialogues were interleaved in every rater's session at random non-adjacent positions:

- `sentinel_clear_improvement`: `case_13__rule_based__pure_cbr_llm__leg_b`, expected R5 = 5, expected R2 = 2.
- `sentinel_clear_flat`: `case_11__socratic__pure_cbr_llm__leg_b`, expected R5 = 1, expected R2 = 3.

Sentinel target ratings were set by inspection of the LLM judge output and cross checked in a colleague pilot before deployment.

---

## 3. Rater recruitment

**Platform.** Prolific (`https://www.prolific.com`), study title "Judging Math Learning in Tutorial Dialogues".

**Filters.**
- Approval rate ≥ 95%.
- Education: Bachelor's degree or higher.
- Subjects studied (multi-select): Mathematics, Statistics, Computer Science, Engineering, Physics, Mathematics Education, Education.
- First language: English.
- Country of residence: UK, Ireland, US, Canada, Australia, New Zealand.
- Age: 22+.

**Target sample.** 20 raters (range 15 to 25), with 4 to 5 raters per dialogue.

**Recruitment waves.**
- Wave 1 launched 1 July 2026, 30 slots.
- Wave 2 (extension on the same Prolific study ID) launched 3 July 2026, 30 additional slots, continuing recruitment until ratings per dialogue reached the target.

**Delivered.** 67 real (non-test) participants, 58 reaching `session_end`, 9 partial dropouts.

**Payment.** Prolific standard fair pay per hour rate, calibrated by a timing pilot.

---

## 4. Instrument

Web-based rating instrument served from `https://tutorial-dialogue-rating.netlify.app/` (source: `validation-study/instrument/index.html`).

**Screens.**
1. Welcome screen with worked example dialogue and scored rationale, AI-generated-content disclosure.
2. Informed consent screen.
3. Two practice trials with feedback:
   - `practice_mid`: expected R5 = 3, R2 = 3.
   - `practice_r2_rule`: expected R5 = 4, R2 = 2.
4. 15 dialogue ratings (13 from the frame + 2 sentinels), Fisher-Yates shuffled per session with sentinels placed at random non-adjacent non-first positions.
5. Session-end summary with optional free-text comment field.

Each dialogue is presented as plain text with the misconception label. Raters apply R5 and R2 items using the same wording and 1 to 5 scales as the LLM judge prompt.

**Backend.** Google Apps Script web app receiving per-event POSTs, appending to a Google Sheet. Live data at `validation-study/data/Responses.csv` (wave 1) and `validation-study/data/Responses2.csv` (cumulative through wave 2).

**Blinding.** Raters were blind to (i) the LLM judge's R5 score for each dialogue, (ii) the persona claim, and (iii) the architecture.

**Pseudonymity.** Each participant received a random 8-character participant code generated with `crypto.getRandomValues`. No name, email, IP address, or Prolific ID is recorded in the research data.

---

## 5. Exclusion rules

### 5.1 Protocol strict exclusion (headline analysis)

A rater is excluded if either:

- **Sentinel rule.** More than one of the four sentinel item scores (R5 and R2 for each of the two sentinels) falls outside its documented pass band. Pass bands:
  - `sentinel_clear_improvement`: R5 ∈ {4, 5}, R2 ∈ {1, 2, 3}.
  - `sentinel_clear_flat`: R5 ∈ {1, 2}, R2 ∈ {2, 3, 4}.
- **Practice rule.** Practice deviation of > 2 on either axis (R5 or R2) in **both** practice trials.

Delivered result: 33 completers retained, 25 completers excluded (mostly on the sentinel rule), average 3.0 raters per frame dialogue.

### 5.2 Post hoc sensitivity variant (robustness check)

Motivated by post hoc inspection of wave 1 sentinel behaviour, in which many raters read the `sentinel_clear_flat` student utterance "it's a triangular based pyramid" as partial trajectory rather than the sustained confusion the LLM judge scored R5 = 1: the `sentinel_clear_flat` R5 pass band is relaxed from {1, 2} to {1, 2, 3}. All other rules unchanged.

Delivered result: 47 completers retained, average 4.1 raters per frame dialogue.

The strict analysis is the headline reported in the manuscript. The sensitivity variant is reported alongside as a robustness observation and explicitly labelled as post hoc.

---

## 6. Analysis structure

### 6.1 Stage 1: within pool reliability

Question: Are humans reliable on the construct?

Statistic: ICC(2,1) and ICC(2, k̄) computed via a two-way random-effects model fit with statsmodels `mixedlm` and crossed variance components (`analysis/human_validation.py::_stage1_icc_variance`) on the sparse rater × dialogue long table. Interpretive bands from Koo and Li (2016).

Gate: ICC(2, k̄) ≥ 0.70 is the pool aggregation criterion.

### 6.2 Stage 2: LLM to human agreement (per dialogue)

Question: Does the LLM judge agree with the pool of human raters?

Statistic: Per-dialogue human mean (across raters with ≥ 2 ratings on that dialogue) vs. the LLM judge single rating. Pearson r, Spearman ρ, and ICC(2,1) (two-way random, absolute agreement, single measure). Bootstrap 95% CIs on all three statistics from 2{,}000 resamples of paired dialogue rows.

Decision bands on the primary Pearson r:
- Strong: r ≥ 0.75.
- Adequate: 0.60 ≤ r < 0.75.
- Partial: 0.40 ≤ r < 0.60.
- Divergence: r < 0.40.

Each band carries a defined reporting posture in the manuscript.

### 6.3 Stage 3: architecture rank preservation

Question: Is the architecture ordering preserved between LLM and human scoring?

Statistic: Per-architecture human R5 mean vs. LLM R5 mean. Spearman ρ and Kendall τ between the five architecture rankings.

Also reported: the primary contrast (hybrid vs. best non-hybrid) as Welch's t and Cohen's d, with remaining hybrid vs. other contrasts under Holm correction at family-wise α = 0.05.

---

## 7. Files deposited

All files are frozen at the time of this deposit.

- `OSF_STUDY_PROTOCOL.md` (this file).
- `sampling_frame/sampling_frame_manifest.json` — 150-dialogue sampling frame with SHA-256 hash.
- `sampling_frame/sampling_frame_manifest.sha256.txt` — hash file.
- `sentinels/sentinel_clear_improvement.json`, `sentinels/sentinel_clear_flat.json` — sentinel dialogues.
- `instrument/INSTRUMENT_SPEC.md` — instrument specification.
- `instrument/index.html` — deployed instrument HTML.
- `instrument/google_apps_script.gs` — backend endpoint.
- `data/Responses.csv` — wave 1 cumulative export.
- `data/Responses2.csv` — wave 1 + wave 2 cumulative export (headline dataset).
- `analysis/human_validation.py` — analysis script (link to GitHub repository).

The corresponding analysis code and released dialogue dataset are at `https://github.com/CCC-NCI/mnemonic-cbr`.

---

## 8. Ethics

The study protocol was submitted to the National College of Ireland Research Ethics Committee on 13 May 2026. No committee decision had been received by the recruitment window (1 and 3 July 2026), and recruitment proceeded on the basis of the study's minimal risk profile (adult crowdworkers, no real student data, pseudonymous, fair pay). A retrospective notification and full study documentation is filed with the ethics committee at the time of manuscript resubmission. This departure is disclosed in the manuscript's §Ethical Considerations and Declarations rather than through omission.

---

## 9. Corresponding manuscript

Janetzko, D. and González-Vélez, H. (2026). "Dovetailing Case Based Reasoning and Large Language Models to Compare Teaching Strategies: A Multi Turn Dialogue Framework using the EEDI Dataset." Manuscript submitted to the International Journal of Artificial Intelligence in Education.

Section references: methodology in §5.5 (Empirical Validation of R5 and R2), results in §6.7 (Human Rater Validation of R5 and R2), ethics in §9.
