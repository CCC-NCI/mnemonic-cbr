# Prolific Rating Instrument: Screen by Screen Specification

Action plan reference: VALIDATION_STUDY_ACTION_PLAN.md §0.7.
Revision history: rev 0 (15 June 2026) initial draft; rev 1 (16 June 2026) revised against comment 1 (verbatim judge text, R2 sentinel coverage, halo break, practice trial, pseudonymous wording, dropped eyesight question, residence filter location, templated time, fixed count justifications); rev 2 (16 June 2026) revised against comment 2: the low R5 sentinel is replaced with a Socratic teacher / persistent misconception dialogue so the pair brackets R2; a second extreme practice trial is added that exercises the R2-from-teacher-only rule; the `[TIMING_PILOT_RESULT]` token is split into `[TIMING_PILOT_MINUTES]` (estimated assignment time) and `[N_DIALOGUES_PER_ASSIGNMENT]` (the design quantity, conversation count); rev 3 (16 June 2026, this version) revised against comment 3: the two sentinels no longer share a misconception or opening student turn (the low R5 sentinel is re-selected from a distinct case); the R2 sentinel pass bands are re-anchored to the co-author calibration-pilot expert consensus rather than to the LLM judge, removing the circularity whereby the gate filtered raters by proximity to the quantity Stage 2 estimates; a per-rater response-constancy exclusion check is added to catch item-specific parking (e.g. R2 held at the modal value) that global straight-liner checks miss; both practice trials are now registered in the OSF deposit. The LLM judge rerun under R2 disambiguation is registered in §6.1 of `OSF_PREREGISTRATION.md`.

The instrument is what a Prolific rater sees from the moment they click the study link until the moment they receive the completion code. It collects R5 (student trajectory) and R2 (cognitive demand) ratings on five turn teacher student dialogues, with two sentinel dialogues interleaved at random positions and a fixed count free text justification subsample.

## Item equivalence with the LLM judge

The point of Stage 2 of the validation study is to enter the LLM judge as one additional rater alongside the human pool on the same items. That move is only valid if the human rater and the LLM judge are responding to the same item. The instrument therefore reproduces the judge's R5 and R2 wording verbatim from `mnemonic-cbr/code/scoring/prompts.py` PASS1_PROMPT, with no per score anchor descriptors (the judge prompt does not contain anchor descriptors) and no editorial gloss. The judge prompt also does not present an "I cannot tell" option; the rater does not see one either. These choices are pre registered (Section 6 of `OSF_PREREGISTRATION.md`).

## Targets

- Expected end to end time per assignment depends on the timing pilot result. The welcome screen reads `about [TIMING_PILOT_MINUTES] minutes` and the bracketed token is replaced by the post pilot recosted figure before the live round opens.
- The assignment length in conversations is a separate token `[N_DIALOGUES_PER_ASSIGNMENT]` set from the recosting decision in `TIMING_PILOT_PROTOCOL.md`. The two tokens are distinct because one is a measured pilot output and the other is a design quantity (raters x dialogues per rater / total dialogues to be rated).
- The rating interface fits on a standard laptop screen at 1280 by 800 without scrolling for the rubric definitions.
- Dialogue presentation order is randomised per rater under a server side seed so rater fatigue does not align with dialogue characteristics.
- Sentinel positions are randomised within the assignment so a rater cannot predict where they will appear.

## Screen 1: Welcome

**Purpose.** Welcome the rater, confirm they are in the right study, and let them self-assess fit on the maths content via a short example dialogue before they commit to the task.

**Content.**

- Study title: "Assessing tutorial dialogues in math".
- One paragraph plain language summary: "You will read about [N_DIALOGUES_PER_ASSIGNMENT] short conversations between a maths teacher and a student. Each conversation has five turns. You will give each conversation two scores on a 1 to 5 scale. The task takes about [TIMING_PILOT_MINUTES] minutes. You can stop at any time."
- A full five-turn example dialogue rendered with the same speaker tags and styling the rater will see throughout, drawn from a misconception not in the rated set (the canonical "1/2 + 1/3 = 2/5" misconception). The example runs S0 to S4 and shows the student arriving at the correct answer 5/6, so the rater sees exactly the shape of dialogue they will be scoring. The example is read only; it is not scored.
- **Scored-example explanation block.** Immediately below the example, the welcome screen states the expected scores (R2 = 4, R5 = 5) and gives a one-paragraph rationale for each. This converts the example from a passive "can you read this?" check into an active "can you score it the way we expect?" check. A would-be guesser whose intuition is far from the explained scoring sees the gap and self-selects out; a legitimately maths-aware rater whose intuition matches the explanation continues with calibrated expectations. This block was added for the Prolific recruitment channel where unfiltered guessers are a non-trivial risk.
- One sentence inviting self-selection: "If the level of maths above feels comfortable, continue. If not, you can return your submission to Prolific without penalty."
- A "Begin" button.

**Field validation.** None. Click only.

**Why a self-selection example here.** The example helps a rater whose maths comfort is below UK secondary level to decline before consent and rating start, without the stigma of failing a screening. It is a fit check rather than an eligibility screen, so it sits before consent in the screen order. The eligibility filter on Screen 3 (UK secondary maths completed in the last ten years) still applies as the formal entry criterion; the example is an additional and softer self-check.

**Budget implication.** Self-selection means some recruited Prolific raters will decline at Screen 1 and not proceed to rating. Under Prolific's fair-pay rules, raters who decline after starting the study are still entitled to a small minimum payment for the time they spent reading the welcome screen. The timing pilot (`TIMING_PILOT_PROTOCOL.md`) records the self-selection decline rate alongside the per-rating time so the recosting decision accounts for it. A 5 to 10 percent decline rate at this screen is the working assumption; if the timing pilot finds a higher decline rate, the main round payment is adjusted to keep total budget within the GBP 400 to 700 envelope.

## Screen 2: Informed consent

**Purpose.** Capture explicit consent before any rating occurs. The wording is the consent text from the NCI ethics application Appendix B.

**Content.**

- One page consent text covering: who the researchers are, the purpose of the study, what the rater will do, payment, voluntary participation, the right to withdraw, data handling, contact details for follow up questions.
- The data handling sentence reads: "Your responses are pseudonymous: only your Prolific ID is recorded, not your name or contact details. Prolific can link the ID back to your account; we cannot."
- Two checkboxes.
  - "I have read and understood the information above."
  - "I consent to participate in this study."
- A "Continue" button, disabled until both checkboxes are ticked.

**Field validation.** Both checkboxes required. Click only after both are ticked.

## Screen 3: Eligibility screening

**Purpose.** Confirm UK secondary maths background.

**Content.** Three questions.

1. "Did you complete UK secondary mathematics?" Yes / No.
2. "When did you complete it?" Within the last 10 years / More than 10 years ago.
3. "How would you rate your current confidence with school-level mathematics, on a 1 to 5 scale? (This does not affect eligibility; it helps us interpret the ratings.)" 1 to 5 radio buttons.

Q3 is a **descriptive covariate**, not a screen-out criterion. It is recorded for exploratory analysis (e.g. sensitivity of LLM-to-human ICC to rater self-rated confidence) and registered in §2.4 of `OSF_PREREGISTRATION.md` as exploratory. Self-rated confidence is well known to interact with the Dunning-Kruger pattern, so using it as a hard filter would risk over-excluding competent low-confidence raters and admitting over-confident weak raters.

**Filters that are NOT enforced here.** UK residence and Prolific approval rate at least 95 percent are enforced via the Prolific platform recruitment filters, not in the instrument. This split is documented in §2.4 of `OSF_PREREGISTRATION.md`.

**Field validation.** All three required. The rater proceeds only if Q1 is Yes and Q2 is "Within the last 10 years". Q3 has no eligibility consequence. Otherwise they see a polite screen out message and return to Prolific.

## Screen 4: Instructions and rubric definitions

**Purpose.** Train the rater on R5 and R2 with the same wording the LLM judge sees in its Pass 1 prompt, so the human pool and the LLM judge are compared on equally defined items.

**Content.** The text below is verbatim from the LLM judge prompt (`mnemonic-cbr/code/scoring/prompts.py` PASS1_PROMPT) with only the surrounding framing added.

> You will read short tutoring dialogues. Each dialogue has five turns: the student speaks first, then the teacher, then the student again, alternating, ending on the student. For each dialogue you give two scores on a 1 to 5 scale.
>
> R2. Cognitive demand: Does the teacher require the student to reason, or just state the answer?
>
> *For R2, judge from what the teacher asks, regardless of how the student responds. A teacher who demands reasoning but is met with a confused student still scores high on R2. A teacher who states the answer but happens to provoke a thoughtful student response still scores low on R2.*
>
> R5. Student trajectory: Does the student's reasoning visibly improve across turns? Consider whether the dialogue plausibly corrects the misconception, judging only from what the student says.
>
> Score each item from 1 (poor) to 5 (excellent). Be strict; reserve 5 for genuinely strong instances.

The italic R2 disambiguation paragraph is the only addition to the LLM judge prompt text. It addresses the halo risk that R5 and R2 are scored on the same screen with the same dialogue in view. It is documented as an instrument level halo break in §6 of `OSF_PREREGISTRATION.md` and the same disambiguation is appended to the LLM judge prompt before the rerun of the LLM judge against the rated dialogues, so the judge and the rater see the same disambiguation. The rerun is captured in `mnemonic-cbr/results/check2_icc_rerun_with_r2_clarification/` (pending) and the original 1,250 dialogue scores are reused if and only if a sensitivity analysis shows the disambiguation does not shift LLM scores by more than 0.3 on either item on a 50 dialogue audit subset.

The instructions screen also flags the practice trial:

> Before you start rating, you will do two short practice trials. Each practice dialogue is followed by the scores we expected, so you can see how the rubric reads in concrete cases. The practice dialogues do not count toward your assignment.

- A "Start practice" button is disabled for the first 90 seconds (the front end timestamps screen entry and disables the button) to prevent skip through reading.

**Field validation.** None. Click only after at least 90 seconds.

## Screen 4a: Practice trials with shown expected scores

**Purpose.** Calibrate the rater against two worked examples before the live ratings begin. Removes the risk that a Stage 1 ICC failure is attributable to lack of training rather than to rubric construct. Two trials rather than one because a single mid range exemplar is a thin calibration for a longitudinal construct: the first trial anchors the midpoint and the second trial sharply exercises the R2-from-teacher-only rule.

**Content.**

The same rating interface as Screen 5 but with the heading "Practice trial X of 2. Not counted."

**Practice trial 1, mid range.** R2 expected 3, R5 expected 3. The dialogue shows a student who begins to apply the right method but does not finish, with a teacher who asks one focused question. On submit, the screen reveals: "Our expected scores are R2 = 3 and R5 = 3. The teacher asks one focused question without demanding multi-step reasoning (R2 mid). The student starts to use the right method but does not finish (R5 mid)." R2 is named before R5 to match the rating-interface order (Comment 1 halo break: R2 above R5).

**Practice trial 2, R2-from-teacher rule.** R2 expected 2, R5 expected 4. The dialogue shows a student who progresses cleanly to a correct final application under a teacher who is mostly stating the rule (the same R5/R2 pattern direction as the high R5 sentinel — student up, teacher demand down — but with a different misconception and a different surface problem so the rater does not learn the sentinel by heart). R5 = 4 rather than 5 because the student rephrases and applies the teacher's correction rather than reaching the rule independently. R5 = 4 still pairs with R2 = 2 to teach the R2-from-teacher rule; the lesson does not depend on R5 being at the extreme. On submit, the screen reveals: "Our expected scores are R2 = 2 and R5 = 4. The teacher is mostly stating the rule rather than asking the student to reason (R2 low). The student's reasoning visibly improves and they correctly apply the rule to their diagram (R5 strong, but not full because they rephrase the teacher's correction rather than reasoning their way there independently). Even when the student's reasoning is good, R2 reads the teacher, not the student." R2 is named before R5 to match the rating-interface order.

Between the two trials and at the end of the second trial, one brief sentence asks the rater to reflect: "If your scores differed by more than one from the expected, take a moment to reread the rubric definitions above." After the second trial the button reads "Begin live ratings".

**Field validation.** R5 and R2 both required on each trial. The rater's practice scores are recorded for both descriptive reporting and for the practice-trial deviation exclusion rule (registered in §3.5 of `OSF_PREREGISTRATION.md`).

## Screen 5: Rating interface (repeated for each dialogue)

**Purpose.** The actual rating, one dialogue at a time.

**Layout (top to bottom).**

1. Progress bar: "Dialogue X of N".
2. Misconception label, one line. Example: "Confuses the order of operations, believes addition comes before multiplication."
3. The five turn dialogue, rendered with speaker tags (S0, T1, S2, T3, S4), each turn on its own block. Teacher and student turns are visually distinguishable but not labelled by persona or architecture.
4. R2 rating block (presented first to weaken the halo from R5 onto R2). Heading "R2. Cognitive demand". Item text verbatim from Screen 4 including the disambiguation paragraph. 1 to 5 radio buttons, no anchor descriptors. Default state: nothing selected.
5. R5 rating block. Heading "R5. Student trajectory". Item text verbatim from Screen 4. 1 to 5 radio buttons, no anchor descriptors. Default state: nothing selected.
6. (Conditional, see "Justification sampling" below) Free text justification box. If the dialogue is one of the rater's selected justification dialogues, the screen shows two single sentence text fields, "In one sentence, what is the main reason for your R5 score?" and "In one sentence, what is the main reason for your R2 score?", each capped at 300 characters and each optional.
7. "Next dialogue" button, disabled until both R5 and R2 are scored.

**Field validation.** R5 and R2 both required. If the justification fields are shown they are optional but the button is enabled only after both R5 and R2 are scored.

**Timing capture.** The front end records the time between dialogue render and the click on "Next dialogue" as the per dialogue rating time. Used for the timing pilot calibration and for the exclusion rule (median rating time outside [30 seconds, 8 minutes]).

## Screen 6: Debrief

**Purpose.** Thank the rater, give the Prolific completion code, provide contact details for follow up questions.

**Content.**

- "Thank you. Your ratings have been recorded."
- "If you have questions about the study, contact dietmar.janetzko@ncirl.ie."
- The Prolific completion code.
- A "Return to Prolific" button.

## Behavioural rules

- **Order randomisation.** Dialogues are shown in an order generated server side from a rater specific seed. Sentinels are inserted at two random positions in the rater's queue with the constraint that neither sentinel is in position 1 (no rater starts with a sentinel) and the two sentinels are not adjacent.
- **No back navigation.** Once the rater submits a dialogue rating they cannot revisit it. This prevents anchoring against earlier ratings within the same rater.
- **Browser locked state.** The session stores the rater's progress so they can resume on accidental reload; deliberate navigation away ends the session and the rater must restart.
- **Attention check display.** Sentinels are shown without any visual cue distinguishing them from rated dialogues. The pass band check is applied server side after submission.

## Justification sampling

A fixed count of ceil(0.2 * n) dialogues per rater are flagged for the justification fields, where n is the rater's total non sentinel dialogue count. The selection is by server side seeded shuffle so the realised coverage is exactly 20 percent rounded up per rater. The earlier per item Bernoulli rule (`Math.random() < 0.2`) is replaced because realised coverage swings well off 20 percent across ~35 items and does not match the "20 percent of each rater's assignments" wording used in the pre registration.

Both R5 and R2 receive a justification prompt on the same dialogues, because Stage 3's qualitative pass on the most divergent dialogues looks at both items.

## Sentinel checks

Two sentinels (`sentinel_clear_improvement.json`, `sentinel_clear_flat.json`) are interleaved at random positions. They are selected so that they share no misconception and no opening student turn, so a blind rater cannot recognise the pair as planted (rev 3). Each sentinel is checked on both R5 and R2 against pass bands documented in `validation-study/sentinels/SELECTION_RATIONALE.md`. The R5 bands are anchored at the scale extremes (the high R5 sentinel resolves the misconception, the low R5 sentinel does not). The R2 bands are anchored at the co-author calibration-pilot expert consensus R2 (± 1), **not** at the LLM judge's R2, so the gate does not exclude raters merely for reading R2 higher than the judge in the mid-range — which is the disagreement Stage 2 is designed to measure. Because the R2 sentinel bands sit in the mid-range and therefore do little attention-checking work on their own, item-specific R2 inattention is caught instead by the response-constancy check below. The full rule is recorded in §3.5 of `OSF_PREREGISTRATION.md`.

**Response-constancy check (rev 3).** A rater who reads R5 attentively but parks R2 at a single modal value (e.g. 3) on every dialogue passes all four sentinel item checks and both practice trials, and is not a *global* straight-liner, so the sentinel+practice+time rule does not catch them. The exclusion rule therefore also excludes any rater who assigns the same value to R2 (or to R5) on at least 90 percent of their non-sentinel dialogues. The 90 percent threshold tolerates a genuinely compressed score distribution while catching constant or near-constant item-specific responding. This is registered in §3.5 of `OSF_PREREGISTRATION.md`.

## Data captured

For each rater:

- Prolific PID, study completion timestamp.
- Per dialogue: dialogue blind ID, R5 score, R2 score, optional R5 free text justification, optional R2 free text justification, rating time in seconds.
- Per sentinel: sentinel ID (clear_improvement or clear_flat), the rater's R5 score, the rater's R2 score, pass band check result on each item.
- Practice trial scores from Screen 4a (not used in analysis but kept for descriptive reporting).
- Eligibility responses from Screen 3.
- Session metadata: order seed, sentinel positions, total task time.

The raw data are exported as CSV and joined to the sampling frame manifest on the blind ID for analysis. The analysis pipeline is `mnemonic-cbr/code/analysis/human_validation.py` (pending implementation, Action plan item 1.1).

## Piloting

The instrument is piloted in three rounds before the timing pilot launches.

1. **Self pilot.** Dietmar completes one full pass, end to end, with a 5 dialogue subset. Goal: catch field validation issues and timing screen oddities.
2. **Co author pilot.** Horacio and Michael each complete one full pass with the same 5 dialogue subset. Goal: confirm rubric definitions read clearly to an outside reader.
3. **Co author calibration pilot.** Horacio and Michael each complete the full 30 to 38 dialogue assignment, treated as expert raters. Their scores are stored separately and used in the descriptive expert micro panel comparison (Action plan §1.3).

None of these three rounds is recruited via Prolific, so none requires ethics approval. They are instrument development.

### Calibration audit on the practice trials

A pre-pilot calibration audit on the practice trial expected scores was run by passing the rev 4 mockup through ChatGPT acting as a rater. Trial 1 (mid-range R5 = 3, R2 = 3) produced a perfect match. Trial 2 (originally R5 = 5, R2 = 2) produced R5 = 3, R2 = 3 on the first pass. After the rater saw the feedback box ("R2 reads the teacher, not the student"), the rater corrected the R2 reading to 2 and stated the lesson back verbatim, confirming that the R2-from-teacher disambiguation lands. The R5 reading was diagnosed by the rater as defensible ("partial but solid correction, not exceptional autonomous conceptual restructuring") — the student in Trial 2 rephrases the teacher's rule rather than reaching it independently, which is canonical R5 = 4 ("strong improvement") rather than R5 = 5 ("clear correction and confident understanding"). The expected score on Trial 2 was therefore lowered from R5 = 5 to R5 = 4 (this revision). The R5 = 4, R2 = 2 pair still teaches the R2-from-teacher lesson because the contrast between strong student progress and low teacher demand is preserved. The Trial 2 dialogue text is unchanged; only the expected score and the feedback wording are updated.

The audit also confirmed the practice-deviation exclusion rule (§3.5 of `OSF_PREREGISTRATION.md`) is correctly scoped: it binds only when both items of a single trial differ from expected by more than one. A defensible R5 = 3 reader on Trial 2 with the old R5 = 5 expected would have failed the rule by chance rather than by inattention. The lowered expected closes this Type I path.

## Implementation options

The action plan allows three implementation paths: Qualtrics, Limesurvey, or a custom static HTML page. The accompanying `instrument_mockup.html` is the custom static implementation, suitable for the three pilot rounds without any third party tooling. For the live Prolific round the instrument needs to be hosted on a stable URL accessible to Prolific raters, with server side capture of per dialogue ratings and sentinel positions. The action plan §0.7 leaves the choice between Qualtrics, Limesurvey, and a custom static page to discussion with Horacio.

## Data submission and partial-completion behaviour

Each rating event is submitted to a Google Apps Script web app as a separate POST as soon as the rater commits to it. The Apps Script appends one row per submission to a Google Sheet in the researcher's Drive. The backend is Apps Script + Sheet rather than Netlify Forms (the original choice) because Netlify's free-tier Akismet spam filter silently discarded rapid per-event POSTs during pre-testing — the script at `instrument/google_apps_script.gs` and the endpoint URL constant `APPS_SCRIPT_URL` in `index.html` document the current setup.

Four event types:

- `practice_rating`: fired when a rater submits a practice trial. Contains `practice_id`, `r5`, `r2`, `expected_r5`, `expected_r2`.
- `dialogue_rating`: fired when a rater clicks "Next dialogue". Contains `dialogue_id`, `sequence_position`, `r5`, `r2`, `rating_time_seconds`. The `sequence_position` field captures the order in which this rater saw the dialogue (Fisher-Yates shuffled at session start), so the order is reconstructable in analysis.
- `session_end`: fired when a rater submits the comments and reaches the success screen. Contains `comments`, `n_practice_completed`, `n_dialogues_completed`, `total_session_seconds`, and a `recovered_events` field bundling any per-event POSTs that failed during the session.
- All events carry `event_type`, `participant_code`, and `submitted_at`. The `participant_code` field is the join key in analysis. The Apps Script adds a server-side `received_at` timestamp on each row, independent of the participant's machine clock.

Each session presents 15 dialogues drawn from the 150 dialogue sampling frame plus the two sentinels. The sampling logic is: shuffle the full 150 dialogue pool, take the first 13, then insert the two sentinels at random non-adjacent positions (never at position 0). This means each rater contributes 13 ratings on a random subset of the 150 frame, distributed approximately uniformly across the pool as rater count grows. At ~40 raters and ~13 frame ratings each, every dialogue in the 150 frame is rated by approximately 4 to 5 raters, matching the registered Stage 2 design.

The randomised dialogue order is the operational guarantee that partial completers contribute unbiased data: a rater who drops out after three dialogues contributes three randomly-drawn dialogues from the 150 dialogue pool, not "the first three of a fixed list" every time.

The full-completion case captures roughly 18 events per rater (two practice plus 15 dialogue plus one session end). The 15 dialogues split as 13 sampling-frame dialogues plus 2 sentinels at random non-adjacent positions. The drop-out case captures everything before the tab close minus the session end. The rare case of a network failure mid-session plus no return is the only loss path, and the localStorage backup queue catches it whenever the rater does return.

Apps Script web apps deployed as "Anyone, even anonymous" with "Execute as: Me" accept unlimited submissions on the free Google tier. No spam filter sits in the path. Submissions arrive in the Google Sheet within seconds and are exported with `File → Download → Comma separated values` for analysis. The sheet does not record participant IP address, user agent, or referrer; only the form-field values the survey explicitly sends. Google's infrastructure logs request metadata for security purposes but does not surface those fields to the script or to the sheet.

## Pass to the analysis pipeline

After the live round the export is a single CSV downloaded from the Google Sheet (`File → Download → Comma separated values`). Each row is one rating event; rows are joined to participants by `participant_code` and to dialogues by `dialogue_id` (which maps to the sampling-frame manifest via the blind ID). The analysis pipeline reconstructs per-rater sessions by grouping on `participant_code`, orders within session by `sequence_position`, and then proceeds to Stage 1, Stage 2, Stage 3 outputs.
