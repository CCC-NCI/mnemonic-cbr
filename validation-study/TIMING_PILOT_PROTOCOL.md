# Timing Pilot Protocol

Action plan reference: VALIDATION_STUDY_ACTION_PLAN.md §0.8 (writing) and §2.1 (execution).

## Purpose

The timing pilot answers two questions before the full Prolific rating round launches.

1. **How long does one rating take?** The Prolific platform requires the study owner to set a per assignment payment that compensates raters at the fair pay rate per hour. Setting that payment without measured timing data either over pays (wasting budget) or under pays (low Prolific approval rates, slow recruitment, fair pay rule violation).
2. **Does the rubric work in the wild?** A handful of free text justifications captured under the same conditions as the main round give an early read on whether non expert UK secondary maths qualified raters can read the rubric the way the LLM judge does.

The timing pilot is not part of the confirmatory analysis. Its outputs only feed the recosting decision and the rubric clarity check before the full round. It is registered on OSF as a preliminary procedure (Section 8 of `OSF_PREREGISTRATION.md`).

## Status

This is a Phase 2 item. It runs on NCI REC approval and not before. Writing this protocol is Phase 0 work and is complete on deposit of this document with the OSF pre registration.

## Configuration

- **n raters.** 2 to 3, recruited via Prolific under the same platform filters as the main round (UK residence, approval rate at least 95 percent, Prolific maths qualification within the last ten years).
- **n dialogues per rater.** 10. The 10 are drawn from the frozen sampling frame manifest as a representative slice across the 15 strata (2 per architecture, alternating low / mid / high R5 bands) plus the two sentinels. So one rater rates 12 items including the two sentinels.
- **Estimated rater time.** Based on the LLM judge's full prompt pass over the same dialogues (about 7 seconds per dialogue for the LLM judge with a target of 75 seconds per dialogue for a human rater) and conservative reading time at 200 words per minute, the expected median per dialogue rating time is 60 to 90 seconds. The pilot tests this estimate against observed behaviour.
- **Per assignment payment.** Set at the Prolific recommended fair pay rate. Working assumption: 90 seconds median per dialogue x 12 dialogues per assignment = 18 minutes of rater work. At the Prolific fair pay rate of approximately GBP 9 per hour (June 2026), that translates to a per assignment payment of about GBP 2.70. The exact value is set in the Prolific dashboard when the pilot launches.

## Outputs

For each pilot rater the instrument captures:

- Median per dialogue rating time, in seconds.
- Sentinel pass / fail (using the bands in §2.3 of `OSF_PREREGISTRATION.md`: clear improvement R5 in {4, 5}; clear flat R5 in {1, 2}).
- Full set of R5 and R2 scores on the 10 non sentinel dialogues plus the 2 sentinels.
- Free text justifications for any dialogue where the instrument's 20 percent random subsample selected the justification field.
- Total session time from instructions to debrief.

Aggregated across 2 to 3 raters:

- Pooled median per dialogue rating time.
- Sentinel pass rate.
- Sample of justifications, read by the authors for evidence that raters understood the rubric the way it is defined in the instrument.

## Decision rules

### Per rating payment for the main round

The pilot's pooled median per dialogue rating time fixes the per assignment payment for the main round. The formula:

```
median_seconds_per_dialogue x dialogues_per_assignment x fair_pay_rate_per_second
```

with the fair pay rate per second derived from the Prolific recommended hourly rate at the time of recruitment.

### Budget envelope

The full round budget is constrained to GBP 400 to 700 (per the NCI ethics application Appendix C). Two main round configurations fit inside that envelope under different observed timings:

| Observed median per dialogue | Dialogues per assignment | Raters per dialogue | Pool size | Estimated total |
|------------------------------:|--------------------------:|-------------------:|---------:|----------------:|
| 60 seconds                    | 38                        | 5                  | 20       | GBP 450 to 500  |
| 90 seconds                    | 38                        | 4                  | 20       | GBP 540 to 600  |
| 120 seconds                   | 30                        | 4                  | 20       | GBP 540 to 600  |
| 150 seconds                   | 24                        | 4                  | 20       | GBP 540 to 600  |
| 180 seconds                   | 20                        | 4                  | 20       | GBP 600 to 700  |

If the observed median per dialogue exceeds 180 seconds, two contingencies kick in.

1. Drop the optional free text justification field from 20 percent of dialogues to 10 percent. Reduces the time burden by approximately 8 seconds per dialogue on average.
2. If still over envelope, reduce the rated dialogue pool from 150 to 120 at the same rater count, accepting the small loss in Stage 1 ICC precision. The reduction is recorded in the manuscript and in the OSF deposit as an amendment.

If the observed median per dialogue is below 60 seconds, recruit two additional raters within the same budget so the per dialogue rater count rises from 4 to 5. Adds approximately GBP 30 to 50 to the total.

### Rubric clarity check

Three thresholds are read against the pilot data and reported in the manuscript as a small descriptive paragraph in §sec:results-human-validation.

- **Sentinel pass rate.** Across the pilot raters and across the two sentinels (4 to 6 sentinel ratings total), at least 80 percent must fall in the pass bands. If below 80 percent, the rubric definitions in Screen 4 are revised before the main round. The revision and any wording change are recorded as an amendment to the OSF deposit.
- **Justification quality.** The authors read every justification captured in the pilot. If three or more justifications give a reason that contradicts the rubric definition (for example, scoring R5 = 5 because "the teacher was nice"), the rubric definitions are rewritten in Screen 4 to address the specific pattern.
- **Drop out.** If any of the 2 to 3 pilot raters does not complete the assignment, the main round assignment length is shortened by 4 dialogues regardless of the timing data.

## Procedure

1. Confirm NCI REC approval.
2. Set up the Prolific study with the pilot configuration: 2 or 3 slots, 10 dialogue plus 2 sentinel assignment, the per assignment payment computed under the working assumption above.
3. Open the study, monitor recruitment until all slots fill (typical 24 to 48 hours).
4. Close the study once all assignments are submitted.
5. Export the rating CSV.
6. Compute pooled median per dialogue rating time. Compute sentinel pass rate. Read every justification.
7. Apply the decision rules above to set the main round configuration.
8. Update the per assignment payment, dialogues per assignment, and rater pool size in the Prolific dashboard for the main round.
9. Launch the main round.

## Recording

The pilot outputs and the resulting main round configuration are recorded in a single markdown file `validation-study/timing_pilot_results.md` written after the pilot completes. That file is deposited with the OSF registration as supplementary material.

## Reproduction note

A reviewer reproducing the timing pilot starts from this protocol and the instrument under `validation-study/instrument/`. The Prolific study configuration (slots, payment, filters) is set in the Prolific dashboard, not in this repository. The raw rating CSV exported from the Prolific dashboard is the input to the recosting decision.
