# Sentinel Dialogue Selection Rationale

Action plan reference: VALIDATION_STUDY_ACTION_PLAN.md §0.6.

Revision history.
Rev 0 (15 June 2026): initial pair covered R5 only.
Rev 1 (16 June 2026): pair revised to cover R5 and R2 simultaneously per comment 1 feedback. Both sentinels sat at low R2 (R2 = 2 and R2 = 1), which left a straight liner at R2 = 1 or R2 = 2 unrecognised.
Rev 2 (16 June 2026): the low R5 sentinel is replaced with one whose R2 sits at the empirical ceiling for R5 = 1 dialogues (R2 = 3, the highest R2 paired with R5 = 1 anywhere in Phase D). The pair now brackets both items at both ends of the scale that the dataset supports. The empirical asymmetry (no R5 = 1 dialogue carries R2 >= 4) is itself a finding and is documented in §2.3 and §6.1 of `OSF_PREREGISTRATION.md`.
Rev 3 (16 June 2026): two changes per comment 3. (a) The two sentinels in rev 2 shared a misconception and a verbatim opening student turn (both were the same 100-minutes-per-hour Hannah/Tori problem). A blind rater seeing the same problem twice can recognise it as a planted item, which defeats the covert attention check and risks anchoring the second reading on the first. The low R5 sentinel is therefore re-selected from a *distinct* Phase D case, sharing neither misconception nor opening turn with the high R5 sentinel. (b) The R2 sentinel pass bands in rev 2 were anchored at the LLM judge's own R2 score (judge ± 1). Because the R2 sentinels sit mid-range, that gate excluded attentive raters who read R2 systematically higher than the judge — exactly the disagreement Stage 2 estimates — biasing the Stage 2 R2 ICC upward by construction. The R2 bands are re-anchored to the co-author calibration-pilot expert consensus (± 1), computed before deposit, and item-specific R2 inattention is policed by a response-constancy check rather than by the mid-range R2 sentinel bands.

Rev 4 (16 June 2026, this version): the distinct low R5 sentinel left as a placeholder in rev 3 is filled in. After querying the Phase D manifest for R5 = 1, R2 at the empirical ceiling (R2 = 3), R4 >= 4, R6 >= 4, persona Socratic or constructive, and case_id distinct from case_13, the strongest unambiguously-flat candidate is `case_11__socratic__pure_cbr_llm__leg_b.json` (misconception "Struggles to visualise nets of 3D shapes"; LLM judge scores R1 = 2, R2 = 3, R3 = 2, R4 = 4, R5 = 1, R6 = 5). The dialogue file has been copied into `sentinels/sentinel_clear_flat.json`, and the placeholder narrative in §"Sentinel 2" below is replaced with the real turn level content. The R2 expert-consensus pass band remains pending the co-author calibration pilot, which freezes the band before deposit.

The two sentinel dialogues are interleaved at random positions in every rater's task so we can verify that the rater is reading the dialogue rather than scoring at random or straight lining one or both items. Because Stages 1, 2, and 3 of the validation study read R5 and R2 together, each sentinel must be unambiguous on both items. The pair is selected so that R5 is bracketed at opposite ends across the two sentinels, and so that the two sentinels share neither a misconception nor an opening student turn (rev 3) — a blind rater must not be able to recognise the pair as planted. R2 is bracketed only across the range the dataset supports (see §"Why the low R5 sentinel does not have R2 >= 4"); the residual R2 attention-checking work is done by the response-constancy check, not by the mid-range R2 sentinel bands.

## Pass-band anchoring (rev 3)

R5 pass bands are anchored at the scale extremes that the sentinels occupy (high R5 sentinel: R5 in {4, 5}; low R5 sentinel: R5 in {1, 2}). These extremes are uncontroversial and human–judge agreement on them is near universal, so a judge-referenced ± 1 band introduces no circularity.

R2 pass bands are **not** anchored at the LLM judge's R2 score. The R2 sentinels sit mid-range (R2 around 2 and around 3), where legitimate human disagreement is largest; anchoring the band at judge ± 1 would exclude attentive raters who read R2 a point higher than the judge, which is the very disagreement Stage 2 is built to detect, and would bias the Stage 2 R2 ICC upward. The R2 bands are instead set at the **co-author calibration-pilot expert consensus R2 (± 1)** for each sentinel. The calibration pilot (Horacio and Michael as expert raters; §Piloting round 3 of `INSTRUMENT_SPEC.md`) runs before deposit and requires no ethics approval, so the consensus R2 values and the resulting bands are frozen into the OSF registration. The numeric R2 bands below are populated from that pilot before the deposit is timestamped.

## Selection procedure

1. **Filter the Phase D manifest for the high R5 sentinel.** Required: R5 = 5 AND a discriminating R2 score AND R4 >= 4 (so the success state is not contaminated by wrong mathematical content). The chosen high R5 sentinel pairs R5 = 5 with R2 = 2 (student progresses despite a directive teacher). The combination R5 = 5, R2 = 5 does not exist in Phase D; the LLM judge never assigned R5 = 5 and R2 = 5 to the same dialogue.

2. **Filter the Phase D manifest for the low R5 sentinel.** Required: R5 = 1 AND R2 as high as the data supports AND R4 >= 4 AND a misconception distinct from the high R5 sentinel's, with a distinct opening student turn (rev 3 constraint). The empirical ceiling for R5 = 1 in Phase D is R2 = 3. There are no dialogues with R5 <= 2 AND R2 >= 4 anywhere in the n = 1,250 manifest. The chosen low R5 sentinel therefore pairs R5 = 1 with R2 = 3, drawn from a case other than the high R5 sentinel's, and the empirical observation is recorded as a methodological note for the LLM judge rerun under R2 disambiguation (§6.1 of the OSF deposit, `mnemonic-cbr/code/experiments/rerun_with_r2_clarification.py`). The concrete source dialogue is `case_11__socratic__pure_cbr_llm__leg_b.json` (LLM judge scores R1 = 2, R2 = 3, R3 = 2, R4 = 4, R5 = 1, R6 = 5; misconception "Struggles to visualise nets of 3D shapes"). Both the JSON file in `validation-study/sentinels/sentinel_clear_flat.json` and the mockup dialogue text in `instrument_mockup.html` now carry this live dialogue rather than a placeholder.

3. **Read the candidates by hand.** For each candidate the five turn dialogue is read in full. The chosen examples are the ones whose R5 trajectory and whose R2 teacher behaviour are visible at the sentence level, so a non expert rater can reach the same score without needing domain expertise.

4. **Set the R2 pass bands from the calibration pilot, not the judge (rev 3).** After selection, Horacio and Michael rate both sentinels on R2 as expert raters in the pre-deposit co-author calibration pilot. The R2 pass band for each sentinel is the expert consensus R2 (± 1). The R5 pass bands remain anchored at the sentinel's scale extreme. All four bands are frozen into the OSF deposit before recruitment.

## Sentinel 1: clear improvement, low cognitive demand (R5 = 5, R2 = 2)

**File:** `sentinel_clear_improvement.json`
**Source path:** `mnemonic-cbr/results/phaseB_smoke_5turn/dialogues/case_13__rule_based__pure_cbr_llm__leg_b.json`
**LLM judge scores:** R1 = 5, R2 = 2, R3 = 4, R4 = 5, R5 = 5, R6 = 5.

**Misconception.** "Answers as if there are 100 minutes in an hour when changing from hours to minutes." Hannah's journey is 3 hours 16 minutes, Tori's is 214 minutes, difference in minutes. Student gets 102, the wrong-conversion answer.

**Trajectory across turns.** Turn 0 (student): wrong answer 102. Turn 1 (teacher, rule based): "you may be assuming there are 100 minutes in an hour. Remember, there are 60 minutes in an hour." Turn 2 (student): "I forgot there's 60 minutes in an hour, not 100. So 3 times 60 is 180? Then add the 16 minutes on top?" Turn 3 (teacher): confirms. Turn 4 (student): "180 plus 16 is 196 minutes for Hannah's journey. Then 214 minus 196 equals 18 minutes." Student arrives at the correct final answer.

**Why R5 = 5 is unambiguous.** Student begins with the wrong method, names the conceptual gap, performs the correct conversion, and arrives at the right numerical answer.

**Why R2 = 2 is unambiguous.** Teacher states rules directly and confirms partial answers rather than asking the student to derive the conversion.

**Pass bands.** R5 in {4, 5}; R2 = calibration-pilot expert consensus R2 (± 1), frozen at deposit (expected {1, 2, 3} if the consensus is 2). The R2 band is set from the expert pilot, not from the LLM judge's R2 = 2, per §"Pass-band anchoring".

## Sentinel 2: clear flat, mid cognitive demand (R5 = 1, R2 = 3), distinct case

**File:** `sentinel_clear_flat.json`
**Source path:** `mnemonic-cbr/results/phaseB_smoke_5turn/dialogues/case_11__socratic__pure_cbr_llm__leg_b.json`
**LLM judge scores:** R1 = 2, R2 = 3, R3 = 2, R4 = 4, R5 = 1, R6 = 5.

**Misconception.** "Struggles to visualise nets of 3D shapes." The student is given a net diagram (a circle touching a sector, which is the net of a cone) and answers "Triangle". The misconception shares neither domain nor opening turn with sentinel 1 (which is an arithmetic time conversion problem).

**Why a distinct case (rev 3 constraint, rev 4 fill).** In rev 2 this sentinel reused the same 100-minutes-per-hour problem as sentinel 1, with an identical opening student turn. A blind rater encountering the same problem twice can recognise it as planted, which defeats the covert attention check and risks anchoring the second reading on the first. Sentinel 2 is therefore drawn from a different misconception (geometry rather than time conversion), a different opening turn ("This is the net of a... I got Triangle" rather than "Hannah's journey to camp is 3 hours and 16 minutes... I got 102"), and a different case_id (case_11 rather than case_13). The conceptual "mirror" of rev 2 (same misconception, opposite teacher behaviour) was visible only to the analyst, never to the blind rater, so nothing the rater could use is lost.

**Trajectory across turns.** Turn 0 (student): wrong answer "Triangle". Turn 1 (teacher, Socratic): "Why do you think the net of a 3D shape involves more than just one face? Can you walk me through how you arrived at the shape you selected?" Turn 2 (student): "I think I was just looking at the pointy bit and thought it looked like a triangle. But now you're saying it has more faces, so... when you unfold a 3D shape, you get multiple flat pieces that come together?" Turn 3 (teacher, Socratic): "Can you think about how the flat pieces you mentioned fit together to create the 3D shape? What do you notice about the relationships between the faces, vertices, and edges?" Turn 4 (student): "Um, so like, the edges have to match up when I fold it? But I'm still not really seeing how a circle and a pointy bit go together. Wouldn't a circle just roll away or something? Like, is the pointy bit the side of the shape and the circle the bottom?"

**Why R5 = 1 is unambiguous.** The student never identifies the 3D shape. Turn 4 ends with two open questions about how the parts fit together and an off-target physical-world distractor ("a circle just roll away"). The terminal state is recognisably the same conceptual gap that turn 0 named, restated less confidently rather than resolved.

**Why R2 sits mid-range.** The Socratic teacher asks genuine reasoning prompts in turn 1 (why does a net involve more than one face? walk me through your reasoning) and turn 3 (how do the faces fit together? what about vertices and edges?) without stating the answer. Cognitive demand is non-trivial even though the student fails to engage with it. The LLM judge's reading is R2 = 3; the **pass band is the calibration-pilot expert consensus R2 (± 1)**, not the judge value, per §"Pass-band anchoring" above.

**Pass bands.** R5 in {1, 2}; R2 = expert-consensus R2 (± 1), frozen at deposit (illustrative band {2, 3, 4} if the consensus is 3).

## Combined attention check

Each rater is exposed to both sentinels at randomised positions. A rater is excluded if **any** of the following holds:

1. **Sentinel failure:** more than one of the four sentinel item checks falls outside its pass band. The R5 bands are the scale extremes; the R2 bands are the calibration-pilot expert consensus (± 1), per §"Pass-band anchoring".
2. **Practice deviation:** on **either** practice trial, both the R5 and the R2 score differ from that trial's registered expected scores by more than 1 point (trial 1 expected R5 = 3, R2 = 3; trial 2 expected R5 = 5, R2 = 2).
3. **Time anomaly:** median per dialogue rating time falls outside [30 seconds, 8 minutes].
4. **Response constancy (rev 3):** the same value is assigned to R2 (or to R5) on at least 90 percent of the rater's non-sentinel dialogues.

The pass band table below uses illustrative R2 bands of {1,2,3} (sentinel 1) and {2,3,4} (sentinel 2); the live bands are the expert-consensus bands frozen at deposit. It is shown against a **global** straight liner (same value on every item):

| Rater scores everything as | Sentinel 1 R5 (pass {4,5}) | Sentinel 1 R2 (illus. {1,2,3}) | Sentinel 2 R5 (pass {1,2}) | Sentinel 2 R2 (illus. {2,3,4}) | Sentinel fails | Caught by |
|----------------------------|:-------------------------:|:----------------------------:|:-------------------------:|:----------------------------:|:-----:|:--------|
| 1                          | fail                      | pass                         | pass                      | fail                         | 2     | sentinel (EXCLUDED) |
| 2                          | fail                      | pass                         | pass                      | pass                         | 1     | constancy (≥90% identical) + practice + time |
| 3                          | fail                      | pass                         | fail                      | pass                         | 2     | sentinel (EXCLUDED) |
| 4                          | pass                      | fail                         | fail                      | pass                         | 2     | sentinel (EXCLUDED) |
| 5                          | pass                      | fail                         | fail                      | fail                         | 3     | sentinel (EXCLUDED) |

The global straight liner at 2 is the one value the sentinel pair alone does not catch (1 sentinel fail); it is caught by the response-constancy check (a global straight liner is 100 percent identical on both items), reinforced by practice deviation and the rating-time band.

**Item-specific parking, not just global straight lining.** The rev 2 catch table considered only *global* straight liners — a rater clicking the same value for everything. It missed the rater who reads R5 attentively but parks R2 at the modal value (e.g. 3): such a rater passes all four sentinel checks (3 lies in both illustrative R2 bands), passes both practice trials, and shows normal timing. The response-constancy check (rule 4) closes this gap by excluding any rater whose R2 (or R5) is ≥ 90 percent identical across the assignment. The corrected claim is therefore: the combined rule (sentinel + practice + time + constancy) catches both global straight lining at every scale value and item-specific parking on either item.

## Why the low R5 sentinel does not have R2 >= 4

A natural reading of comment 2 would be that the low R5 sentinel should sit at R2 = 5: a teacher who demands genuine multi step reasoning while the student stays confused to the end. Phase D does not contain such a dialogue. There is no R5 <= 2 AND R2 >= 4 dialogue anywhere in the n = 1,250 manifest. This empirical asymmetry has two readings:

- **Mechanism.** The LLM judge's R2 reading may be partly conditioned on student response. If the student fails to engage, the judge tends to rate R2 lower than it would have if the same teacher questions had been met with student engagement. This would mean the original LLM judge prompt suffers from exactly the halo the rev 1 R2 disambiguation was added to address on the human side.

- **Phase D distribution.** In this case base, five turn dialogues ending on the student, the typical pattern when R2 is high is that the student at least partially engages, which raises R5 above 1. A dialogue with sustained high cognitive demand and zero student engagement may simply be empirically rare in this design.

Both readings motivate the LLM judge rerun under the same R2 disambiguation the human raters receive. The rerun is registered as a methodological commitment in §6.1 of `OSF_PREREGISTRATION.md`. The R5 = 1, R2 = 3 sentinel chosen here is the strongest dual extreme bracketing available in the pre rerun data; if the rerun produces R5 = 1, R2 = 4 dialogues, the sentinel pair will be reviewed and amended via the OSF amendment mechanism before the live round opens.

## Reproducibility

Both sentinels are persisted as JSON in this directory and are identifiable by their source paths under `mnemonic-cbr/results/phaseB_smoke_5turn/dialogues/`. The high R5 sentinel is `case_13__rule_based__pure_cbr_llm__leg_b.json`; the low R5 sentinel is `case_11__socratic__pure_cbr_llm__leg_b.json`. The pass band rules are encoded in the OSF deposit Section 3.5 and in the analysis pipeline `mnemonic-cbr/code/analysis/human_validation.py` (Action plan item 1.1, pending). A reviewer can verify the selection by inspecting the LLM judge scores in either file and by reading the turn level dialogue against the rationale above. The R2 expert-consensus pass band remains the one parameter that is filled at deposit from the co-author calibration pilot rather than at sentinel selection.
