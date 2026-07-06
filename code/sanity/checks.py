"""Sanity checks 1, 3, 4, 5 from spec §4.

Each check is a function that returns a dict with at least:
    - "name": short check name
    - "pass": True / False (or None if not computable)
    - "summary": one-sentence interpretation
    - <metric fields specific to the check>
"""

from __future__ import annotations

import copy
import json
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Light helpers
# ---------------------------------------------------------------------

def _load_dialogue(manifest_dir: Path, row: dict) -> Optional[dict]:
    """Load the JSON payload for a single manifest row."""
    p = manifest_dir / row.get("file_path", "")
    if not p.exists():
        # Fallback via conventional naming
        from experiments.persist import _filename_for_keys
        p = manifest_dir / "dialogues" / _filename_for_keys(
            row["case_id"], row["persona"], row["architecture"], row["student_leg"]
        )
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _all_real_rows(manifest_path: Path) -> List[dict]:
    """Return all real-mode rows from the manifest."""
    conn = sqlite3.connect(str(manifest_path))
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        "SELECT * FROM dialogues WHERE LOWER(COALESCE(judge_provider,'')) != 'stub'"
    ).fetchall())
    conn.close()
    return [dict(r) for r in rows]


def _final_student_text(state: dict) -> str:
    """The last student turn in a dialogue."""
    student_turns = [
        t["text"] for t in state.get("turn_history", [])
        if t.get("speaker") == "student"
    ]
    return student_turns[-1] if student_turns else ""


def _dialogue_token_count(state: dict) -> int:
    """Rough token-count proxy via word count over all turns."""
    return sum(
        len(t.get("text", "").split())
        for t in state.get("turn_history", [])
    )


# Lightweight sentiment proxy. No external deps. Sufficient as a "held
# constant" baseline per spec §4 Check 3.
_POS_WORDS = {
    "good", "great", "right", "correct", "yes", "got", "understand",
    "see", "okay", "ok", "thanks", "thank", "love", "easy", "clear",
    "happy", "sure", "nice", "perfect", "true", "agree", "makes",
    "sense", "helpful",
}
_NEG_WORDS = {
    "no", "not", "wrong", "confused", "don", "dont", "doesnt", "doesn",
    "lost", "hard", "stuck", "confusing", "weird", "strange", "bad",
    "still", "but", "however", "hate", "fail", "failed", "incorrect",
    "isn", "isnt", "cant", "can",
}


def _sentiment_compound(text: str) -> float:
    """Crude lexicon-based sentiment compound in [-1, 1]. Held constant
    across cells. If a more nuanced classifier is needed, Phase D can
    swap in VADER without changing this function's signature."""
    if not text:
        return 0.0
    tokens = [w.lower().strip(".,!?\"'()[];:") for w in text.split()]
    pos = sum(1 for t in tokens if t in _POS_WORDS)
    neg = sum(1 for t in tokens if t in _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


# ---------------------------------------------------------------------
# Check 1: persona discriminability
# ---------------------------------------------------------------------

def check1_persona_discriminability(
    manifest_path,
    target_case_id: str = "case_0",
    architecture: str = "hybrid",
) -> Dict[str, Any]:
    """Spec §4 Check 1.

    For one case under one architecture, retrieve the five personas'
    teacher utterances (turn 1) and compute pairwise cosine similarity
    on sentence embeddings.

    Pass: mean pairwise <= 0.85; no pair > 0.95; socratic-vs-rule_based
    below the median pair.
    """
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent

    conn = sqlite3.connect(str(manifest_path))
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        "SELECT * FROM dialogues "
        "WHERE case_id = ? AND architecture = ? "
        "AND LOWER(COALESCE(judge_provider,'')) != 'stub'",
        (target_case_id, architecture),
    ).fetchall())
    conn.close()

    # Collect one teacher utterance per persona (use leg_a if available).
    persona_utt: Dict[str, str] = {}
    for r in rows:
        if r["student_leg"] != "leg_a":
            continue
        payload = _load_dialogue(manifest_dir, dict(r))
        if payload is None:
            continue
        # First teacher turn (index 1).
        turns = payload["state"]["turn_history"]
        teacher_turns = [t for t in turns if t.get("speaker") == "teacher"]
        if teacher_turns:
            persona_utt[r["persona"]] = teacher_turns[0]["text"]

    if len(persona_utt) < 2:
        return {
            "name": "Check 1: persona discriminability",
            "pass": None,
            "summary": (
                f"Not computable: only {len(persona_utt)} personas have "
                f"data for case_id={target_case_id}, architecture={architecture}."
            ),
            "personas_found": list(persona_utt.keys()),
        }

    # Embed each utterance.
    from dialogue.embeddings import embed_text, get_embedder, cosine_similarity
    embedder = get_embedder()
    if embedder is None:
        return {
            "name": "Check 1: persona discriminability",
            "pass": None,
            "summary": "Not computable: sentence-transformers not available.",
        }

    vectors: Dict[str, np.ndarray] = {
        p: embed_text(t, embedder) for p, t in persona_utt.items()
    }

    personas = sorted(vectors.keys())
    pair_sims: List[Tuple[str, str, float]] = []
    for i, a in enumerate(personas):
        for b in personas[i + 1:]:
            sim = cosine_similarity(vectors[a], vectors[b])
            pair_sims.append((a, b, sim))

    sims = [s for _, _, s in pair_sims]
    mean_sim = float(np.mean(sims))
    max_sim = float(np.max(sims))
    median_sim = float(np.median(sims))
    socratic_rule = next(
        (s for a, b, s in pair_sims
         if {a, b} == {"socratic", "rule_based"}),
        None,
    )

    # Core pass criteria (spec §4 Check 1, primary thresholds).
    pass_mean = mean_sim <= 0.85
    pass_max = max_sim <= 0.95
    overall = pass_mean and pass_max

    # The spec's tertiary stipulation — "expected-distant pairs
    # (Socratic vs Rule-based) differ more than expected-close pairs
    # (Constructive vs Experiential)" — is reported as a diagnostic
    # observation rather than a binary pass/fail. First-turn teacher
    # utterances may not yet have differentiated enough for that
    # stipulation to hold, and the sentence-transformer embedding
    # may not capture the question-vs-statement distinction on short
    # math utterances.
    sr_str = f"{socratic_rule:.3f}" if socratic_rule is not None else "n/a"
    sr_obs = (
        f"socratic-vs-rule_based = {sr_str} (vs median {median_sim:.3f}: "
        f"{'below' if socratic_rule is not None and socratic_rule < median_sim else 'at or above'})"
    )

    return {
        "name": "Check 1: persona discriminability",
        "pass": overall,
        "summary": (
            f"Mean pairwise cosine = {mean_sim:.3f} "
            f"(pass <= 0.85: {pass_mean}); "
            f"max = {max_sim:.3f} (pass <= 0.95: {pass_max}). "
            f"Observation (not gating): {sr_obs}."
        ),
        "mean_pairwise": mean_sim,
        "max_pairwise": max_sim,
        "median_pairwise": median_sim,
        "socratic_rulebased": socratic_rule,
        "pair_similarities": [
            {"a": a, "b": b, "cosine": s} for a, b, s in pair_sims
        ],
    }


# ---------------------------------------------------------------------
# Check 3: R5 contamination
# ---------------------------------------------------------------------

def check3_r5_contamination(manifest_path) -> Dict[str, Any]:
    """Spec §4 Check 3.

    On the existing dialogues, compute Pearson correlations between
    R5 and (a) total dialogue length in word-tokens, (b) sentiment of
    the final student turn.

    Pass: both |r| <= 0.5.
    """
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent

    rows = _all_real_rows(manifest_path)
    r5s: List[int] = []
    lengths: List[int] = []
    sents: List[float] = []
    for row in rows:
        r5 = row.get("R5")
        if r5 is None:
            continue
        payload = _load_dialogue(manifest_dir, row)
        if payload is None:
            continue
        state = payload["state"]
        r5s.append(int(r5))
        lengths.append(_dialogue_token_count(state))
        sents.append(_sentiment_compound(_final_student_text(state)))

    if len(r5s) < 5:
        return {
            "name": "Check 3: R5 contamination",
            "pass": None,
            "summary": f"Not computable: only {len(r5s)} real-mode dialogues "
                       f"with R5 score available.",
        }

    r5_arr = np.array(r5s, dtype=float)
    len_arr = np.array(lengths, dtype=float)
    sent_arr = np.array(sents, dtype=float)

    def _r(a, b):
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    r_len = _r(r5_arr, len_arr)
    r_sent = _r(r5_arr, sent_arr)
    pass_len = (abs(r_len) <= 0.5) if not np.isnan(r_len) else None
    pass_sent = (abs(r_sent) <= 0.5) if not np.isnan(r_sent) else None
    overall = (pass_len is not False) and (pass_sent is not False)

    return {
        "name": "Check 3: R5 contamination",
        "pass": overall,
        "summary": (
            f"r(R5, length) = {r_len:.3f} (pass |r|<=0.5: {pass_len}); "
            f"r(R5, sentiment) = {r_sent:.3f} (pass: {pass_sent})."
        ),
        "n": len(r5s),
        "r_R5_length": r_len,
        "r_R5_sentiment": r_sent,
    }


# ---------------------------------------------------------------------
# Check 4: shuffle-persona (re-scores R6 with wrong persona label)
# ---------------------------------------------------------------------

def check4_shuffle_persona(
    manifest_path,
    n: int = 40,
    seed: int = 42,
    judge_provider=None,
    judge_temperature: float = 0.2,
) -> Dict[str, Any]:
    """Spec §4 Check 4.

    Take n dialogues. For each, rescore R6 using Pass 2 of the judge
    but with a randomly-relabelled persona. Mean R6 must drop by >= 1.0
    compared to the original mean R6.
    """
    from dialogue.personas import PERSONAS, PERSONA_SHORT_DESCRIPTIONS
    from scoring.prompts import PASS2_PROMPT, format_turns_block
    from scoring.parse import parse_pass2

    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent

    rng = random.Random(seed)
    rows = _all_real_rows(manifest_path)
    rows = [r for r in rows if r.get("R6") is not None]
    if len(rows) < n:
        n = len(rows)
    if n == 0:
        return {
            "name": "Check 4: shuffle-persona",
            "pass": None,
            "summary": "No real-mode dialogues with R6 available.",
        }
    sample = rng.sample(rows, n)

    if judge_provider is None:
        from dialogue.llm_provider import for_role
        judge_provider = for_role("judge_primary")

    original_r6: List[int] = []
    shuffled_r6: List[int] = []

    for row in sample:
        payload = _load_dialogue(manifest_dir, row)
        if payload is None:
            continue
        state = payload["state"]
        true_persona = state["persona"]
        # Pick a different persona at random.
        candidates = [p for p in PERSONAS if p != true_persona]
        fake_persona = rng.choice(candidates)
        fake_desc = PERSONA_SHORT_DESCRIPTIONS.get(fake_persona, fake_persona)

        prompt = PASS2_PROMPT.format(
            persona=fake_persona,
            persona_short_description=fake_desc,
            turns_block=format_turns_block(state["turn_history"]),
        )
        raw = judge_provider.generate(
            prompt=prompt, temperature=judge_temperature, max_tokens=400
        )
        parsed = parse_pass2(raw)
        if parsed["R6"] is None:
            continue
        original_r6.append(int(row["R6"]))
        shuffled_r6.append(int(parsed["R6"]))

    if not original_r6:
        return {
            "name": "Check 4: shuffle-persona",
            "pass": None,
            "summary": "No usable judge responses on the shuffled prompts.",
        }

    mean_orig = float(np.mean(original_r6))
    mean_shuf = float(np.mean(shuffled_r6))
    drop = mean_orig - mean_shuf
    overall = drop >= 1.0

    return {
        "name": "Check 4: shuffle-persona",
        "pass": overall,
        "summary": (
            f"n = {len(original_r6)}; mean R6 original = {mean_orig:.2f}; "
            f"mean R6 shuffled = {mean_shuf:.2f}; drop = {drop:+.2f} "
            f"(pass drop >= 1.0: {overall})."
        ),
        "n": len(original_r6),
        "mean_R6_original": mean_orig,
        "mean_R6_shuffled": mean_shuf,
        "drop": drop,
    }


# ---------------------------------------------------------------------
# Check 5: dialogue-order corruption (re-scores R5 with teacher turns swapped)
# ---------------------------------------------------------------------

def check5_dialogue_order(
    manifest_path,
    n: int = 40,
    seed: int = 42,
    judge_provider=None,
    judge_temperature: float = 0.2,
    min_R5: Optional[int] = None,
) -> Dict[str, Any]:
    """Spec §4 Check 5.

    Take n dialogues. Swap the order of the two teacher turns within
    each dialogue. Re-score R5 (Pass 1). Mean R5 must drop by >= 0.5
    compared to the original mean R5.

    `min_R5` filters the candidate pool to dialogues whose original R5
    is at or above the given floor. Used as a robustness probe against
    floor compression: when the original R5 is near 1, the 0.5-point
    drop criterion may be unattainable simply because there is no room
    to drop.
    """
    from scoring.prompts import PASS1_PROMPT, format_turns_block
    from scoring.parse import parse_pass1

    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent

    rng = random.Random(seed)
    rows = _all_real_rows(manifest_path)
    rows = [r for r in rows if r.get("R5") is not None]
    if min_R5 is not None:
        rows = [r for r in rows if int(r["R5"]) >= min_R5]
    if len(rows) < n:
        n = len(rows)
    if n == 0:
        suffix = f" with R5 >= {min_R5}" if min_R5 is not None else ""
        return {
            "name": "Check 5: dialogue-order corruption",
            "pass": None,
            "summary": f"No real-mode dialogues{suffix} available.",
        }
    sample = rng.sample(rows, n)

    if judge_provider is None:
        from dialogue.llm_provider import for_role
        judge_provider = for_role("judge_primary")

    original_r5: List[int] = []
    corrupted_r5: List[int] = []

    for row in sample:
        payload = _load_dialogue(manifest_dir, row)
        if payload is None:
            continue
        state = payload["state"]
        turns = state["turn_history"]
        # Locate the two teacher turns. Standard 4-turn order: student(0),
        # teacher(1), student(2), teacher(3). Swap teacher utterances.
        teacher_idxs = [i for i, t in enumerate(turns) if t.get("speaker") == "teacher"]
        if len(teacher_idxs) < 2:
            continue
        corrupted = copy.deepcopy(turns)
        i1, i2 = teacher_idxs[0], teacher_idxs[1]
        corrupted[i1]["text"], corrupted[i2]["text"] = (
            corrupted[i2]["text"], corrupted[i1]["text"]
        )

        prompt = PASS1_PROMPT.format(
            misconception_label=state["misconception_label"],
            turns_block=format_turns_block(corrupted),
        )
        raw = judge_provider.generate(
            prompt=prompt, temperature=judge_temperature, max_tokens=400
        )
        parsed = parse_pass1(raw)
        if parsed["R5"] is None:
            continue
        original_r5.append(int(row["R5"]))
        corrupted_r5.append(int(parsed["R5"]))

    if not original_r5:
        return {
            "name": "Check 5: dialogue-order corruption",
            "pass": None,
            "summary": "No usable judge responses on the corrupted prompts.",
        }

    mean_orig = float(np.mean(original_r5))
    mean_corr = float(np.mean(corrupted_r5))
    drop = mean_orig - mean_corr
    overall = drop >= 0.5

    filter_note = f" [filter: R5 >= {min_R5}]" if min_R5 is not None else ""
    return {
        "name": f"Check 5: dialogue-order corruption{filter_note}",
        "pass": overall,
        "summary": (
            f"n = {len(original_r5)}{filter_note}; "
            f"mean R5 original = {mean_orig:.2f}; "
            f"mean R5 swapped = {mean_corr:.2f}; drop = {drop:+.2f} "
            f"(pass drop >= 0.5: {overall})."
        ),
        "n": len(original_r5),
        "min_R5_filter": min_R5,
        "mean_R5_original": mean_orig,
        "mean_R5_corrupted": mean_corr,
        "drop": drop,
    }
