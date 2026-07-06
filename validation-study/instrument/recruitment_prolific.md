# Prolific study setup

## Study title (shown in the participant's list)

```
Rating short maths tutoring dialogues — UK secondary / US middle-school maths
```

## Study description (longer text shown to participants before they accept)

```
You will read 15 short tutoring dialogues between a teacher and a
student about school-level maths topics (algebra, geometry,
fractions). For each dialogue you will give two simple 1-to-5
ratings: how much the teacher demands reasoning, and how much
the student's understanding visibly improves across the
conversation.

The task takes about 15 to 20 minutes, including two short
practice trials with feedback up front so the rating criteria
make sense in concrete cases.

A worked example with our expected scores is shown on the
welcome screen so you can decide whether the maths level and
the rating logic are a fit for you before continuing. If they
are not a fit, please return your submission with no penalty.

The dialogues are computer-generated; this is disclosed on the
welcome and consent screens before any rating starts.

Eligibility: comfortable with school-level algebra, geometry,
and fractions (UK secondary, roughly US middle-school to early
high-school). Adults aged 18 or over.

Participation is pseudonymous: a random 8-character code is
recorded with your ratings; no name, email, or IP address is
written into the research data.
```

## Prolific platform filters (set in the Prolific dashboard)

| Filter | Setting |
|--------|---------|
| Approval rate | ≥ 95 % |
| Education | Bachelor's degree or higher |
| Subjects studied (multi-select) | Mathematics, Statistics, Computer Science, Engineering, Physics, Mathematics Education, Education |
| First language | English |
| Country of residence | UK, Ireland, US, Canada, Australia, New Zealand |
| Age | 22+ |

Prolific shows the eligible pool size after each filter. Aim for an eligible pool of at least 500 so the 25-completer target is reachable in 24 to 48 hours.

## Estimated time and payment

- Estimated completion time: 20 minutes (set this in the Prolific dashboard)
- Reward: at Prolific's fair-pay rate. For 20 minutes at the UK fair-pay floor (£9/hour), that is £3.00 per completer. Prolific will display the per-hour rate alongside this figure so participants can see it is fair pay.
- Total study cost for 25 completers: roughly £80 to £100 including Prolific's service fee (33% on top of participant payments).

## Sample size and stopping rule

- Initial recruitment: 25 completers.
- If after 25 completers the practice-trial deviation rate is high (more than 30 % outside the registered band), launch a second small batch with the same study but tighten the Prolific platform filter further (e.g. add "Mathematics" as a required subject rather than optional).
- Close recruitment at 25 valid completer-equivalent sessions (distinct `participant_code` with `session_end` row in the Google Sheet, after applying the practice and sentinel exclusion rules registered in §3.5 of `OSF_PREREGISTRATION.md`).

## URL to paste into the Prolific study setup

```
https://tutorial-dialogue-rating.netlify.app/
```

## Completion code

Prolific requires participants to enter a completion code at the end of the task to confirm completion. The current `index.html` shows the participant their random 8-character code on the success screen. Use that code as the completion confirmation: in the Prolific dashboard set the completion mode to "URL with manual code entry" and instruct participants to enter the 8-character code shown at the end of the task.

If you prefer Prolific's automated redirect mechanism instead, the survey would need a small JavaScript patch on the success screen to redirect to `https://app.prolific.co/submissions/complete?cc=YOUR_PROLIFIC_COMPLETION_CODE`. Tell me if you want that change.

## After the study runs

- Export the Google Sheet to CSV (File → Download → Comma separated values).
- Run the analysis pipeline `mnemonic-cbr/code/analysis/human_validation.py` (pending implementation, Action plan item 1.1).
- Report Stage 1, 2, 3 outcomes in the manuscript §sec:results-human-validation.
