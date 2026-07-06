# Human validation study

Full documentation and data for the two-wave Prolific human rater validation study
described in the parent manuscript. The same materials are deposited on the Open
Science Framework at https://osf.io/ae8qm/ with a timestamped deposit record.

## Contents

- `STUDY_PROTOCOL.md` — Study protocol and analysis plan (the load-bearing document).
- `sampling_frame/` — Frozen 150-dialogue sampling frame with SHA-256 hash.
- `sentinels/` — Two hand-picked sentinel dialogues used for rater attention checks.
- `instrument/` — The deployed rating instrument (`index.html`) and the Google Apps
  Script backend that received per-event POSTs (`google_apps_script.gs`), plus the
  screen-by-screen specification (`INSTRUMENT_SPEC.md`) and the recruitment texts
  used across channels.
- `data/` — Cumulative rating dataset:
  - `Responses.csv` — Wave 1 export.
  - `Responses2.csv` — Wave 1 + Wave 2 cumulative export (the file the analysis
    pipeline reads).
- `TIMING_PILOT_PROTOCOL.md` — Timing pilot protocol used to calibrate per-rating
  payment.
- `Letter-to-NCI-Ethics-Committee.pdf` — The retrospective notification sent to the
  NCI Research Ethics Sub-Committee on 5 July 2026. Retained for audit trail.
- `NCI Ethics Application Form_ Human Participants Feb 2026.pdf` — The NCI
  guidelines document the retrospective notification is anchored against.

## Reproducing the analysis

The Prolific ratings are analysed by `code/analysis/human_validation.py` at the
repository root. See the main README for the end-to-end reproduction sequence.

## Pseudonymity

All participant identifiers in `Responses.csv` and `Responses2.csv` are random
8-character codes generated with `crypto.getRandomValues`. No names, email
addresses, IP addresses, or Prolific IDs are recorded in the research data.
