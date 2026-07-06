"""Sentence-embedding helper for retrieval.

Spec reference: IMPLEMENTATION_PLAN §6 (retrieval quality concern flagged
in the Phase A smoke output). The 4-feature legacy retrieval was
essentially random across topics; replacing it with sentence embeddings
of `misconception + construct + question text` makes retrieval
semantically meaningful.

Design choices:

- **Local model only.** We use sentence-transformers' all-MiniLM-L6-v2
  (~80MB, 384-dim). No API spend; full reproducibility. Anthropic does
  not expose embeddings; OpenAI does but is paid and adds another
  provider dependency.
- **Graceful fallback.** If sentence-transformers isn't installed,
  embed_text() returns None and retrieval falls back to feature-based
  similarity. This means the code is importable in any environment;
  the user installs sentence-transformers when they want embedding-based
  retrieval.
- **In-memory cache** keyed by text. Phase A/B-plumbing have very few
  unique texts; the cache prevents re-encoding the same case at every
  similarity comparison. Disk caching is deferred to Phase D where the
  case count and run count justify it.

Usage:

    from dialogue.embeddings import get_embedder, embed_case_ext, embed_query
    embedder = get_embedder()           # None if sentence-transformers absent
    vec = embed_case_ext(case_ext, embedder)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Default model — small, fast on CPU, decent retrieval quality.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Singleton-ish state. Set on first successful load; None if unavailable.
_EMBEDDER = None
_EMBEDDER_LOAD_FAILED = False
_CACHE: dict = {}  # text -> np.ndarray


if TYPE_CHECKING:
    from dialogue.state import CaseExt
    from cbr.mnemonic_augmentation import Case


def get_embedder(model_name: str = DEFAULT_MODEL_NAME):
    """Return a loaded SentenceTransformer model, or None if unavailable.

    The first call loads the model (downloads on first run; cached
    afterwards). Subsequent calls return the same instance.
    """
    global _EMBEDDER, _EMBEDDER_LOAD_FAILED
    if _EMBEDDER is not None:
        return _EMBEDDER
    if _EMBEDDER_LOAD_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; retrieval will use "
            "the legacy feature-based similarity. To enable "
            "embedding-based retrieval: pip install sentence-transformers"
        )
        _EMBEDDER_LOAD_FAILED = True
        return None
    try:
        _EMBEDDER = SentenceTransformer(model_name)
        logger.info(f"Loaded sentence-transformer model: {model_name}")
        return _EMBEDDER
    except Exception as e:
        logger.warning(
            f"Failed to load {model_name}: {e}. "
            f"Falling back to feature-based similarity."
        )
        _EMBEDDER_LOAD_FAILED = True
        return None


def _build_text_for_case(case_ext) -> str:
    """Concatenate the case fields used for retrieval embedding.

    Order matters slightly for transformer models (early tokens
    weighted slightly more). Misconception first; topic next; question
    last.
    """
    intervention = getattr(case_ext.case, "intervention", None) or {}
    parts = []
    misconception = case_ext.misconception or intervention.get("misconception_name", "")
    if misconception:
        parts.append(str(misconception))
    construct = intervention.get("construct_name", "")
    if construct:
        parts.append(f"Topic: {construct}")
    if case_ext.problem_text:
        # Single-line and trim — embedding models have token limits.
        problem = " ".join(case_ext.problem_text.split())
        parts.append(f"Question: {problem[:300]}")
    return ". ".join(parts) if parts else "(no text)"


def _build_text_for_legacy_case(case) -> str:
    """Same as _build_text_for_case but for a bare legacy Case object
    that doesn't have a CaseExt wrapper (e.g., a query Case constructed
    inside TeacherGenerator._retrieve)."""
    intervention = getattr(case, "intervention", None) or {}
    parts = []
    misconception = getattr(case, "misconception", "")
    if misconception:
        parts.append(str(misconception))
    construct = intervention.get("construct_name", "") if isinstance(intervention, dict) else ""
    if construct:
        parts.append(f"Topic: {construct}")
    # For query cases without problem text, intervention may still carry the
    # construct; otherwise we fall back to misconception only.
    return ". ".join(parts) if parts else "(no text)"


def embed_text(text: str, embedder=None) -> Optional[np.ndarray]:
    """Embed a single text. Returns None if no embedder is available.

    Cached in-memory by text.
    """
    if not text:
        return None
    if text in _CACHE:
        return _CACHE[text]
    if embedder is None:
        embedder = get_embedder()
    if embedder is None:
        return None
    vec = embedder.encode(text, convert_to_numpy=True, show_progress_bar=False)
    # Normalise to unit length once so cosine becomes a dot product.
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    _CACHE[text] = vec
    return vec


def embed_case_ext(case_ext, embedder=None) -> Optional[np.ndarray]:
    """Embed a CaseExt for retrieval. Returns None if unavailable."""
    return embed_text(_build_text_for_case(case_ext), embedder)


def embed_legacy_case(case, embedder=None) -> Optional[np.ndarray]:
    """Embed a bare Case object for retrieval. Returns None if unavailable."""
    return embed_text(_build_text_for_legacy_case(case), embedder)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between unit-normalised vectors (becomes a
    dot product). Returns 0.0 for incompatible shapes."""
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def is_available() -> bool:
    """True iff embedding-based retrieval is currently enabled."""
    return get_embedder() is not None


def cache_size() -> int:
    """Number of unique texts currently cached. For diagnostics."""
    return len(_CACHE)


def clear_cache() -> None:
    """Drop the in-memory cache. Useful for unit tests."""
    _CACHE.clear()
