# mnemonic-cbr

**Dovetailing Case-Based Reasoning and Large Language Models to Compare Teaching Strategies: A Multi-Turn Dialogue Framework using the EEDI Dataset**

_A framework for comparing AI tutoring strategies on matched student misconceptions before classroom deployment. IJAIED 2026 submission by Dietmar Janetzko and Horacio González-Vélez, National College of Ireland._

- **Manuscript:** currently under revision at the *International Journal of Artificial Intelligence in Education*.
- **Human validation study:** OSF deposit at [https://osf.io/ae8qm/](https://osf.io/ae8qm/) — study protocol, sampling frame with SHA-256 hash, sentinel dialogues, deployed instrument HTML, and cumulative rating dataset.
- **Contact:** Dietmar Janetzko `dietmar.janetzko@ncirl.ie`.

---

## What the framework does

For each combination of one EEDI misconception case, one pedagogical persona, one system architecture, and one student LLM leg, the framework generates a five-turn teacher–student dialogue and scores it on six rubric items with an external LLM judge. Because the misconception is held constant across pedagogical conditions, the framework isolates the effect of pedagogy from case-level variation. Two of the six rubric items (R5 terminal reasoning state, R2 cognitive demand) are anchored against human raters in a two-wave Prolific rating round.

**What the framework claims and what it does not.** It produces *conditional pre-classroom evidence* about which tutoring designs warrant piloting in a classroom setting. It does not measure real student learning, and the paper does not claim that any specific architectural contrast is a decisive result. Where the observed contrast is small (as it is on the primary outcome R5), the honest reading is spelled out in the manuscript and repeated below.

---

## Design and scale

The reported run comprises 1,250 five-turn dialogues:

| factor        | levels                                                                        | n |
|:--------------|:------------------------------------------------------------------------------|:--:|
| case          | EEDI misconception cases (stratified across misconception families, seed 42) | 25 |
| architecture  | baseline, pure_ai, pure_cbr_llm, pure_cbr_tpl, hybrid                        | 5  |
| persona       | traditional, Socratic, constructive, experiential, rule-based                | 5  |
| student LLM leg | GPT-4o-mini (OpenAI), Claude Haiku 3.5 (Anthropic)                          | 2  |

A larger replication with 150 cases (up to 22,500 dialogues) is in progress and will be released with follow-up work.

**Teacher rendering LLM:** GPT-3.5-turbo. **Rubric judge:** Claude Sonnet, with an inter-judge check against Gemini 2.5 Pro (secondary) and GPT-4o (cross-family confirmation).

---

## Rubric

Six items scored on 1–5 scales. R5 is the primary outcome and R2 is the secondary anchored item; both are validated against human raters. R1, R3, R4, R6 are reported as LLM judge scores anchored against the construct grounding of MQI and RTOP.

| item | measures                                | reporting role       |
|:-----|:----------------------------------------|:---------------------|
| R1   | Misconception engagement                | LLM only             |
| R2   | Cognitive demand                        | Anchored (secondary) |
| R3   | Scaffolding fit                         | LLM only             |
| R4   | Domain accuracy (hallucination monitor) | LLM only             |
| R5   | Terminal reasoning state                | **Anchored (primary)** |
| R6   | Strategy fidelity                       | LLM only             |

---

## Key results

### Architectural comparison (LLM judge, mixed-effects model)

Statistical modelling uses a mixed-effects Architecture × Persona model with case and student leg as crossed random effects. Random effects account for 43% of R5 variance (ICC(case) = 0.205; ICC(leg) = 0.223).

R5 mean by architecture (five turn, terminal reasoning state):

| architecture   | R5 mean | n   |
|:---------------|:-------:|:---:|
| hybrid         | 2.92    | 250 |
| pure_cbr_llm   | 2.80    | 249 |
| pure_ai        | 2.79    | 249 |
| baseline       | 2.71    | 250 |
| pure_cbr_tpl   | 2.12    | 250 |

**The honest reading.** All four LLM-using architectures cluster within 0.14 points of one another and dominate the no-LLM template floor (`pure_cbr_tpl` at 2.12). Hybrid leads the next-best LLM architecture at Cohen's *d* = 0.09, which is trivial by Cohen (1988) conventions. What is clearly established is that any LLM-based tutor beats the template baseline; adding retrieval on top of an LLM gives only a small extra edge on R5. Architectural effects on the pedagogical quality items R1, R2, R3 are much larger (χ²(4) between 243.75 and 344.83 under the mixed model).

### Inter-judge reliability (Check 2)

50-dialogue stratified subset dual-scored by two independent LLM judges from different model families. Primary secondary judge: Gemini 2.5 Pro. Cross-family confirmation: GPT-4o.

| item | ICC(2,1) vs Gemini 2.5 Pro | 95% CI       | ICC(2,1) vs GPT-4o | 95% CI       |
|:----:|:--------------------------:|:------------:|:------------------:|:------------:|
| R5   | **0.715**                  | [0.55, 0.83] | 0.667              | [0.48, 0.80] |
| R2   | 0.684                      | [0.50, 0.81] | 0.432              | [-0.10, 0.76]|
| R1   | 0.610                      | [0.40, 0.76] | 0.383              | [-0.10, 0.71]|
| R3   | 0.490                      | [0.25, 0.67] | 0.404              | [-0.10, 0.74]|
| R4   | 0.620                      | [0.42, 0.76] | 0.438              | [0.12, 0.66] |
| R6   | 0.682                      | [0.50, 0.81] | 0.698              | [0.46, 0.83] |

Five of six items reach the *adequate* band or better against Gemini 2.5 Pro. R5 sits at the boundary of *strong*. GPT-4o reproduces the pattern on the items the framework relies on most (R5, R6).

### Human validation study (Prolific rating round)

Two-wave recruitment on Prolific in July 2026, using the same study ID across waves. Sixty-seven adult crowdworkers were delivered, 58 reaching `session_end`. Pseudonymous 8-character participant codes with no personally identifying information recorded in the research data.

**Under the protocol strict exclusion rule** (more than one sentinel item outside its pass band, or practice deviation greater than 2 on both practice trials → excluded): 33 raters retained, 3.0 raters per frame dialogue, 114 dialogues rated by at least 2 humans.

**Under a post-hoc sensitivity variant** (relaxes the `clear_flat` sentinel R5 pass band from {1, 2} to {1, 2, 3}, on the empirical observation that many raters read the student's line "it's a triangular based pyramid" as partial progress rather than sustained confusion): 47 raters retained, 4.1 raters per frame dialogue, 137 dialogues rated.

| statistic | R5 (strict) | R5 (sensitivity) | R2 (strict) | R2 (sensitivity) |
|:----------|:-----------:|:----------------:|:-----------:|:----------------:|
| Pearson r | 0.55        | 0.56             | 0.66        | 0.71             |
| Spearman ρ | 0.52       | 0.53             | 0.68        | 0.75             |
| ICC(2,1)  | 0.44        | 0.42             | 0.44        | 0.46             |
| n dialogues | 114       | 137              | 114         | 137              |
| decision band* | partial | partial          | adequate    | adequate         |

*Decision bands (paper protocol, adapted from [Koo and Li 2016](https://doi.org/10.1016/j.jcm.2016.02.012)): strong ≥ 0.75; adequate 0.60 – 0.74; partial 0.40 – 0.59; divergence < 0.40. These are *validation* thresholds (can the LLM judge stand in for a human?), not effect-detection thresholds. By Cohen's (1988) standard conventions for correlation strength, both r = 0.55 and r = 0.71 are large effects.

**Architectural rank preservation on R5.** Under the sensitivity pool, Spearman ρ between per-architecture human means and LLM means is **0.60**. Endpoints agree under both scales (`pure_cbr_tpl` last, `pure_cbr_llm` rank 2). The `hybrid`/`pure_ai` swap at the top is within a tenth of a point on either scale and reads as a tie rather than a reversal. Under the stricter exclusion rule the rank correlation collapses to ρ = 0.10 because per-architecture n ≈ 80 leaves the mid-ranked architectures bunched; the manuscript therefore reports the sensitivity ρ as the summary for architectural claims and the strict figure alongside it.

**Cross-family per-leg check.** On the OpenAI student leg alone (GPT-4o-mini, family independent from the Anthropic judge), `hybrid`, `baseline`, and `pure_cbr_llm` tie for top within 0.02 points, `pure_ai` follows, and `pure_cbr_tpl` sits at the floor. On the Anthropic leg (Claude Haiku 3.5), `hybrid` leads outright. `hybrid` tops or ties for top on both legs; `pure_cbr_tpl` bottoms both (Spearman ρ = 0.667 across the five architectures between the two legs). The absolute R5 scores run about one point higher on the Anthropic leg — a family severity offset — but the architectural conclusion is a rank claim and is preserved on the family-independent leg.

### What this means in practice

1. The framework offers a way to compare AI tutoring designs without needing to run an expensive classroom trial every time.
2. The simulation can point developers and researchers toward the tutoring designs worth taking into a real classroom, and away from those that are not.

---

## Reproduction

### Environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Dependencies of note: `sentence-transformers` (retrieval), `statsmodels` (mixed-effects), `pingouin` (ICC), `pandas`, `scipy`.

### Dataset

Download the EEDI dataset from Kaggle: [alejopaullier/eedi-external-dataset](https://www.kaggle.com/datasets/alejopaullier/eedi-external-dataset). Place `all_train.csv` under `data/`.

### API keys

Set at least `OPENAI_API_KEY` (teacher renderer, GPT-4o-mini student leg) and `ANTHROPIC_API_KEY` (Claude Haiku student leg, Claude Sonnet judge). For the inter-judge Check 2 reproduction, also set `GEMINI_API_KEY`.

```bash
export OPENAI_API_KEY="…"
export ANTHROPIC_API_KEY="…"
export GEMINI_API_KEY="…"
```

### End-to-end reproduction

```bash
# 1. Run the 1,250-dialogue Phase B study (25 cases × 5 arch × 5 persona × 2 legs)
python code/experiments/run_phaseB_smoke.py --n-cases 25 --max-turns 5

# 2. Sanity check battery + student grounding A/B gate
python code/experiments/run_sanity_battery.py

# 3. Inter-judge Check 2 (50-dialogue subset against Gemini 2.5 Pro and GPT-4o)
python code/experiments/run_check2.py --secondary gemini --max-tokens 4096
python code/experiments/run_check2.py --secondary gpt4o

# 4. Manuscript tables and figures (mixed-effects models, per-cell means, Cohen's d)
python code/analysis/mixed_effects.py \
  --manifest results/phaseB_smoke_5turn/manifest.sqlite \
  --out-dir results/phaseB_smoke_5turn/analysis/mixed_effects
python code/experiments/run_phaseD_analysis.py \
  --manifest results/phaseB_smoke_5turn/manifest.sqlite \
  --out results/phaseB_smoke_5turn/analysis

# 5. Human validation analysis (needs Responses2.csv from OSF or the Google Sheet export)
python code/analysis/human_validation.py \
  --responses validation-study/data/Responses2.csv \
  --frame validation-study/sampling_frame/sampling_frame_manifest.json \
  --llm-db results/phaseB_smoke_5turn/manifest.sqlite \
  --out-dir results/human_validation
```

Each script writes to its own subdirectory and produces CSV, Markdown, and LaTeX table outputs.

---

## Repository layout

```
mnemonic-cbr/
├── README.md
├── LICENSE.txt
├── requirements.txt
├── code/
│   ├── cbr/                       # case model, EEDI loader, retrieval
│   ├── dialogue/                  # teacher, student, persona, five-turn loop
│   ├── scoring/                   # RubricScorer, judge prompts, JSON parsing
│   ├── analysis/
│   │   ├── mixed_effects.py       # mixed-effects two-way with crossed random effects
│   │   ├── human_validation.py    # Stage 1/2/3 Prolific analysis pipeline
│   │   ├── aggregate.py           # per-cell, per-arch, per-persona means
│   │   ├── anova.py               # legacy naive ANOVA (kept for reference)
│   │   ├── effect_sizes.py        # Cohen's d, contrast tables
│   │   └── export.py              # CSV/Markdown/LaTeX writers
│   └── experiments/
│       ├── run_phaseB_smoke.py    # main 1,250-dialogue runner
│       ├── run_sanity_battery.py  # Checks 1, 3, 4, 5 orchestrator
│       ├── run_check2.py          # inter-judge ICC against Gemini/GPT-4o
│       ├── run_phaseD_analysis.py # manuscript tables and figures
│       ├── sample_validation_subset.py  # OSF sampling frame builder
│       └── compare_student_modes.py     # student grounding A/B gate
└── results/
    └── phaseB_smoke_5turn/
        ├── manifest.sqlite        # scored dialogues (25 cases × 5 × 5 × 2 = 1,250)
        ├── dialogues/             # per-cell JSON dialogue records
        └── analysis/tables/       # manuscript-ready tables (CSV, MD, LaTeX)
```

The `validation-study/` folder in the OSF deposit contains the sampling frame manifest, sentinel dialogues, deployed instrument HTML, Google Apps Script backend, cumulative response CSV, and the study protocol.

---

## Human validation study documentation

The full protocol, sampling frame with SHA-256 hash, sentinel dialogues, deployed instrument, and cumulative response dataset are deposited on the Open Science Framework at [https://osf.io/ae8qm/](https://osf.io/ae8qm/). The deposit is a *timestamped study documentation record posted at the time of manuscript resubmission*. It is not a pre-registration in the technical sense — data collection preceded deposit — and the paper describes it that way to preserve honest audit.

Practical facts about the study as run:

- **Recruitment.** Two waves on Prolific (1 and 3 July 2026) on the same study ID, using identical filters, instrument, and payment.
- **Filters.** Adults 22+, English first language, Bachelor's degree or higher, approval rate ≥ 95%, subjects studied include Mathematics / Statistics / Computer Science / Engineering / Physics / Mathematics Education / Education.
- **Instrument.** Deployed at `tutorial-dialogue-rating.netlify.app`. Screen 1 welcome + AI disclosure; Screen 2 informed consent; two practice trials with feedback; 15 dialogue ratings (13 from the frame + 2 sentinels at non-adjacent non-first positions, Fisher–Yates shuffled per session); optional free-text comment.
- **Pseudonymity.** Random 8-character participant codes generated with `crypto.getRandomValues`. No name, e-mail, IP address, or Prolific ID recorded in the research data.
- **Payment.** Prolific standard fair-pay hourly rate, calibrated by a timing pilot.
- **Backend.** Google Apps Script web-app receiving per-event POSTs, appending to a Google Sheet, then exported to `validation-study/data/Responses2.csv` (the file the analysis pipeline reads).

Ethics disclosure: the study protocol was submitted to the National College of Ireland Research Ethics Sub-Committee on 13 May 2026. No committee decision was received by the recruitment window. Recruitment proceeded on the basis of the study's minimal-risk profile under manuscript resubmission timeline constraints. A retrospective notification was filed on 5 July 2026 requesting confirmation of the study's categorisation under §3.3 (Exemption from Full Ethical Review) of the January 2026 NCI Guidelines. The manuscript's §Ethical Considerations and Declarations discuss this openly.

---

## Costs

Approximate costs for the reported run at late-2026 API rates:

- **Dialogue generation (1,250 dialogues at 5 turns each):** approximately USD 15–25 with the specified model mix (GPT-3.5-turbo teacher, GPT-4o-mini + Claude Haiku 3.5 students).
- **Judge scoring (2 passes × 1,250 dialogues with Claude Sonnet):** approximately USD 8–12.
- **Inter-judge Check 2 (50 dialogues × 2 secondary judges):** approximately USD 3–5.
- **Human validation (67 crowdworkers at Prolific fair-pay):** approximately GBP 220 including platform fees.

Full end-to-end reproduction of the study costs on the order of USD 40 in API calls plus the Prolific budget for the human validation study.

---

## Citation

If you use this framework or the human validation dataset in your work, please cite:

```bibtex
@article{janetzko2026dovetailing,
  title   = {Dovetailing Case-Based Reasoning and Large Language Models to Compare Teaching Strategies: A Multi-Turn Dialogue Framework using the EEDI Dataset},
  author  = {Janetzko, Dietmar and Gonz{\'a}lez-V{\'e}lez, Horacio},
  journal = {International Journal of Artificial Intelligence in Education},
  year    = {2026},
  note    = {Under revision. Human validation study protocol and data deposited on OSF at https://osf.io/ae8qm/.}
}
```

---

## Acknowledgements

Supported in part by the Erasmus+ programme of the European Union under grant agreement No. 101140316 (Digital4Sustainability, [digital4sustainability.eu](https://digital4sustainability.eu)). Thanks to the EEDI project team and to Kaggle for making the misconception-annotated dataset available under CC0. Thanks to the Prolific rater pool who contributed the human validation ratings, and to Horacio González-Vélez at the Cloud Competency Centre, NCI, for the methodological framing that led to the two-anchor validity approach.

The manuscript's `§Use of Large Language Models` declaration names the specific model versions used as research instruments (teacher renderer, student LLMs, primary judge, secondary judges) and confirms that all scientific claims were authored and verified by the human authors.

---

_Last updated: 6 July 2026._
