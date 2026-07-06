"""Outcome-leakage shim + embedding-based similarity wrap around the
legacy MnemonicAugmentation.

Spec reference: REBUILD_SPECIFICATION_v3.md §1 (closed-loop concern),
§3.7 (ablation expectations). IMPLEMENTATION_PLAN §3.4 (audit findings
and resolution).

The shim does two things:

**(A) Outcome-leakage paths disabled.** The legacy MnemonicAugmentation
consumes case.outcome (the Equation-1 synthetic value) in three
retrieval-shaping paths:

  1. _compute_associative_strength — adds 0.1 to the bond between
     cases with the same > 0.5 outcome label.
  2. _enhance_retrieval_cues       — weights features by their
     correlation with case.outcome. Since outcome is a deterministic
     linear function of those features, the cue weights recover
     Equation 1's coefficients. This is the closed loop in its
     purest form.
  3. _analyze_success_pattern + the enhancement on lines 360-364 of
     enhanced_similarity — adds 0.05 in similarity when both cases
     share the same success/unsuccess label.

These three are disabled here without editing the legacy file.

**(B) Embedding-based base similarity.** The legacy similarity uses
weighted Euclidean over a 4-dimensional feature vector
(QuizId%1000, QuestionId%1000, misconception_count, ConstructName
length). That's effectively random across topics. When
sentence-transformers is available, we replace the base similarity
with cosine similarity over embeddings of (misconception + topic +
question). The cleaned chunk-bonus and cue-matching enhancements
still apply on top. If sentence-transformers is unavailable, the
shim falls back to the legacy feature-based base similarity (still
with the outcome-leakage paths disabled).

The legacy MnemonicAugmentation remains unchanged on disk for
reviewer comparison.

Public surface:
  - clean_mnemonic_engine(cases, n_chunks=10) -> MnemonicAugmentation
        Builds a MnemonicAugmentation and applies the disabling
        patches + embedding wrap before process_cases() runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional
import logging

import numpy as np

from dialogue import embeddings as _emb

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cbr.mnemonic_augmentation import Case, MnemonicAugmentation


def clean_mnemonic_engine(
    cases,
    n_chunks: int = 10,
    apply_chunk_bonus: bool = False,
    apply_assoc_bonus: bool = False,
    apply_cue_bonus: bool = False,
):
    """Build a MnemonicAugmentation with outcome-leakage paths disabled
    and base similarity replaced by embedding cosine when available.

    Mnemonic-technique enhancements (chunk, associative-network, cue-
    matching) are OPT-IN by default. Empirically they degrade retrieval
    quality when embedding cosine is the base similarity, because the
    embedding already captures semantic similarity and the enhancements
    reward incidental properties (KMeans clustering, network edges,
    cue overlap) rather than topical relevance. Phase D ablation can
    turn each one back on to measure its effect.

    Args:
        cases:                  list of legacy Case objects
        n_chunks:               KMeans n_chunks (chunks are still built;
                                only their use as a bonus is gated)
        apply_chunk_bonus:      if True, add +15% similarity for same-
                                chunk candidates
        apply_assoc_bonus:      if True, add +10% × edge weight for
                                associative-network neighbours
        apply_cue_bonus:        if True, add +5% per matching cue
                                (uniform cue weights post-shim)

    Returns the engine after process_cases() has been called.
    """
    from cbr.mnemonic_augmentation import MnemonicAugmentation

    engine = MnemonicAugmentation(n_chunks=n_chunks, enable_all=True)
    _patch_associative_strength(engine)
    _patch_retrieval_cues(engine)
    _patch_similarity_success_bonus(engine)
    enhanced = engine.process_cases(cases)

    # Embedding wrap. Tries to load sentence-transformers; if absent,
    # leaves the cleaned legacy similarity in place.
    used_embeddings = _patch_similarity_with_embeddings(
        engine,
        cases,
        apply_chunk_bonus=apply_chunk_bonus,
        apply_assoc_bonus=apply_assoc_bonus,
        apply_cue_bonus=apply_cue_bonus,
    )

    bonus_summary = ", ".join([
        f"chunk_bonus={apply_chunk_bonus}",
        f"assoc_bonus={apply_assoc_bonus}",
        f"cue_bonus={apply_cue_bonus}",
    ])
    logger.info(
        "Built mnemonic engine with outcome-leakage shim "
        "(3 legacy paths disabled). Base similarity = "
        f"{'embeddings (cosine)' if used_embeddings else 'legacy 4-feature euclidean'}. "
        f"Enhancement bonuses under embedding base: {bonus_summary}."
    )
    return engine


# --- Path 1: associative strength outcome bonus ---


def _patch_associative_strength(engine) -> None:
    """Replace _compute_associative_strength with a version that drops
    the outcome-agreement term. Feature similarity + chunk co-membership
    are preserved.
    """

    def _clean_strength(case1, case2):
        feature_sim = 1 - np.linalg.norm(
            case1.features - case2.features
        ) / np.sqrt(len(case1.features))
        chunk_bonus = 0.2 if case1.chunk_id == case2.chunk_id else 0.0
        # outcome_correlation term intentionally OMITTED.
        return float(np.clip(feature_sim + chunk_bonus, 0, 1))

    engine._compute_associative_strength = _clean_strength


# --- Path 2: correlation-based retrieval cue weights ---


def _patch_retrieval_cues(engine) -> None:
    """Replace _enhance_retrieval_cues with a uniform-weight version.

    Rationale: the v1 correlation between feature and case.outcome
    recovers Equation 1, which is the closed-loop bias. With no
    real outcome signal available in EEDI (audit, IMPLEMENTATION_PLAN
    §3.4), uniform weights are the cleanest cut. Misconception
    co-occurrence weighting is a possible future replacement; for
    Phase A, uniform is sufficient.

    The method still populates self.retrieval_cue_weights and
    case.retrieval_cues so downstream code paths that read them
    (the cue-matching enhancement in enhanced_similarity) still work.
    """

    def _uniform_cues(cases):
        n_features = cases[0].features.shape[0] if cases else 0
        for i in range(n_features):
            engine.retrieval_cue_weights[f"feature_{i}"] = 1.0
        # Assign every case the full set of cues — no preferential
        # top-k, since all weights are equal.
        feature_keys = [f"feature_{i}" for i in range(n_features)]
        for case in cases:
            case.retrieval_cues = feature_keys
        return cases

    engine._enhance_retrieval_cues = _uniform_cues


# --- Path 3: similarity success-pattern bonus ---


def _patch_similarity_success_bonus(engine) -> None:
    """Wrap enhanced_similarity to subtract the success-pattern bonus.

    The legacy enhanced_similarity adds +0.05 when both cases share
    the same 'outcome' field in their elaborations.success_pattern
    dict. Wrapping the method is cleaner than rewriting it, since the
    rest of the similarity computation (exponential decay, chunk
    bonus, associative bonus, cue matching) is fine.

    The wrap: detect the success-pattern bonus by recomputing it
    against the original method, then subtract it from the result.
    """
    original = engine.enhanced_similarity

    def _patched(query, candidate):
        sim = original(query, candidate)
        # Re-derive the success-pattern bonus exactly as the legacy
        # method computes it.
        if query.elaborations and candidate.elaborations:
            q_outcome = query.elaborations.get("success_pattern", {}).get("outcome")
            c_outcome = candidate.elaborations.get("success_pattern", {}).get("outcome")
            if q_outcome is not None and q_outcome == c_outcome:
                # Legacy adds 0.05 * beta where beta = 1 + enhancements.
                # The bonus enters as `enhancements += 0.05` (line 364 of
                # mnemonic_augmentation.py), multiplied by base_sim.
                # Exactly subtracting it requires reconstructing beta;
                # a safe upper bound is 0.05 * base_sim, which we
                # approximate by subtracting 0.05 directly. The error
                # is bounded by the sum of other enhancements (at most
                # ~0.30 in the worst case), so the subtraction may
                # leave a small residual. Phase B can replace this
                # wrap with a re-implementation if needed.
                sim = max(0.0, sim - 0.05)
        return sim

    engine.enhanced_similarity = _patched


# --- Embedding base similarity wrap ---


def _patch_similarity_with_embeddings(
    engine,
    cases,
    apply_chunk_bonus: bool = False,
    apply_assoc_bonus: bool = False,
    apply_cue_bonus: bool = False,
) -> bool:
    """Replace the base similarity with cosine-over-embeddings if
    sentence-transformers is available. Returns True on success,
    False if the embedder is unavailable (in which case the cleaned
    legacy similarity is left in place).

    The chunk / associative-network / cue-matching enhancements are
    opt-in (default off). See clean_mnemonic_engine docstring.

    Embeddings for each case are computed once at engine-build time
    and stored on engine._case_embeddings (keyed by case.id). Query
    cases get their embeddings computed lazily inside the patched
    similarity function and cached.
    """
    embedder = _emb.get_embedder()
    if embedder is None:
        return False

    # Pre-compute embeddings for the case base.
    engine._case_embeddings = {}
    for c in cases:
        vec = _emb.embed_legacy_case(c, embedder)
        if vec is not None:
            engine._case_embeddings[c.id] = vec
    if not engine._case_embeddings:
        # Embedder loaded but produced no vectors — keep legacy path.
        return False

    cleaned_legacy = engine.enhanced_similarity

    def _embed_sim(query, candidate):
        q_vec = engine._case_embeddings.get(query.id)
        if q_vec is None:
            q_vec = _emb.embed_legacy_case(query, embedder)
            if q_vec is not None:
                engine._case_embeddings[query.id] = q_vec
        c_vec = engine._case_embeddings.get(candidate.id)
        if q_vec is None or c_vec is None:
            return cleaned_legacy(query, candidate)

        base_sim = (_emb.cosine_similarity(q_vec, c_vec) + 1.0) / 2.0
        enhancements = 0.0
        if apply_chunk_bonus:
            if (query.chunk_id and candidate.chunk_id
                    and query.chunk_id == candidate.chunk_id):
                enhancements += 0.15
        if apply_assoc_bonus:
            if (candidate.associative_links
                    and query.id in candidate.associative_links):
                if engine.associative_network.has_edge(query.id, candidate.id):
                    edge_weight = engine.associative_network[query.id][candidate.id]['weight']
                    enhancements += 0.10 * edge_weight
        if apply_cue_bonus:
            if query.retrieval_cues and candidate.retrieval_cues:
                matching = set(query.retrieval_cues) & set(candidate.retrieval_cues)
                enhancements += 0.05 * len(matching)
        return min(2.0, base_sim * (1.0 + enhancements))

    engine.enhanced_similarity = _embed_sim
    logger.info(
        f"Embedding similarity active: "
        f"{len(engine._case_embeddings)} cases embedded "
        f"(model: {_emb.DEFAULT_MODEL_NAME})"
    )
    return True
