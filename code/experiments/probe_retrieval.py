"""Probe retrieval ranking for a chosen query case.

Prints the top-N retrievals with cosine similarity, chunk membership
(query vs candidate), and whether the same-chunk bonus fired. Useful
for diagnosing why retrieval is or isn't returning topically relevant
cases.

Usage:

    # Default — probe case_0 (the BIDMAS question) against the first 500 cases
    python code/experiments/probe_retrieval.py

    # Probe a different case with a bigger case base
    python code/experiments/probe_retrieval.py --query-case 2 --case-base-size 1000

    # Show the top 20 retrievals
    python code/experiments/probe_retrieval.py --top 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _setup_path() -> None:
    here = Path(__file__).resolve().parent
    code_dir = here.parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


_setup_path()


from dialogue import embeddings as _emb                            # noqa: E402
from dialogue.retrieval import clean_mnemonic_engine               # noqa: E402
from experiments.eedi_loader import load_n_usable_cases            # noqa: E402
from experiments._run_logger import (                               # noqa: E402
    run_with_logging,
    set_log_path,
    utc_stamp,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--data", default="data/all_train.csv")
    p.add_argument(
        "--case-base-size", type=int, default=500,
        help="How many EEDI cases to load into the retrieval base",
    )
    p.add_argument(
        "--query-case", type=int, default=0,
        help="Index into the case base for the query case (default: case_0)",
    )
    p.add_argument(
        "--top", type=int, default=10,
        help="How many top retrievals to print (default 10)",
    )
    p.add_argument(
        "--n-chunks", type=int, default=8,
        help="KMeans n_chunks (default 8 — matches run_phaseB_smoke.py)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_log_path(
        Path("results/probe_retrieval/logs")
        / f"probe_retrieval_{utc_stamp()}.txt"
    )

    print(f"Loading {args.case_base_size} cases from {args.data}...")
    cases_ext = load_n_usable_cases(n=args.case_base_size, filepath=args.data)
    legacy_cases = [ce.case for ce in cases_ext]
    print(f"  loaded {len(cases_ext)}")
    print()

    print(f"Building cleaned mnemonic engine (n_chunks={args.n_chunks})...")
    engine = clean_mnemonic_engine(legacy_cases, n_chunks=args.n_chunks)
    print()

    if args.query_case >= len(cases_ext):
        print(f"ERROR: query_case index {args.query_case} >= case base size",
              file=sys.stderr)
        return 2
    query_ext = cases_ext[args.query_case]
    query_case = query_ext.case
    print(f"Query: {query_case.id}")
    print(f"  misconception: {query_case.misconception}")
    print(f"  construct:     {query_case.intervention.get('construct_name','')}")
    print(f"  chunk:         {query_case.chunk_id}")
    print()

    embedder = _emb.get_embedder()
    q_vec = _emb.embed_legacy_case(query_case, embedder) if embedder else None
    print(f"Embedding model: "
          f"{'active — ' + _emb.DEFAULT_MODEL_NAME if q_vec is not None else 'NONE (fallback to features)'}")
    print()

    # Compute the engine's full similarity for ranking, plus the raw cosine
    # and the chunk-bonus-eligibility flag for diagnostic display.
    rows = []
    for c in legacy_cases:
        if c.id == query_case.id:
            continue
        # Raw cosine (independent of any enhancements).
        if q_vec is not None:
            c_vec = engine._case_embeddings.get(c.id) if hasattr(engine, "_case_embeddings") else None
            if c_vec is None:
                c_vec = _emb.embed_legacy_case(c, embedder)
            cosine_raw = (_emb.cosine_similarity(q_vec, c_vec) + 1.0) / 2.0
        else:
            cosine_raw = float("nan")
        # Engine similarity (with all enhancements).
        engine_sim = engine.enhanced_similarity(query_case, c)
        same_chunk = (query_case.chunk_id is not None
                      and query_case.chunk_id == c.chunk_id)
        rows.append((engine_sim, cosine_raw, same_chunk, c))

    rows.sort(key=lambda r: r[0], reverse=True)
    top = rows[: args.top]

    print(f"Top {args.top} retrievals (ranked by engine similarity):")
    print()
    print(f"{'rank':>4}  {'engine_sim':>10}  {'cosine':>7}  "
          f"{'chunk':>10}  {'topical?':>9}  topic / misconception")
    print("-" * 100)
    for i, (eng, cos, same, c) in enumerate(top, start=1):
        chunk_disp = f"{c.chunk_id}" + (" *same*" if same else "")
        topic = (c.intervention.get("construct_name") or "")[:38]
        misc = (c.misconception or "")[:38]
        print(f"  {i:>2}.  {eng:>10.4f}  {cos:>7.4f}  "
              f"{chunk_disp:>16}  -  {topic} | {misc}")
    print()

    # Diagnostic: how often does same-chunk bonus matter?
    same_chunk_in_top = sum(1 for (_, _, same, _) in top if same)
    print(f"Of top {args.top}: {same_chunk_in_top} are in the SAME chunk as the query.")
    print(f"Query chunk id: {query_case.chunk_id}")
    # How does same-chunk-only ranking compare?
    cosine_ranked = sorted(rows, key=lambda r: r[1], reverse=True)[: args.top]
    cosine_ids = {c.id for (_, _, _, c) in cosine_ranked}
    engine_ids = {c.id for (_, _, _, c) in top}
    overlap = len(cosine_ids & engine_ids)
    print(f"Overlap between engine-ranked top-{args.top} and pure-cosine top-{args.top}: {overlap}")
    print()
    if overlap < args.top:
        print("Pure-cosine top retrievals NOT in engine top (i.e. demoted by enhancements):")
        for (_, cos, same, c) in cosine_ranked:
            if c.id not in engine_ids:
                topic = (c.intervention.get("construct_name") or "")[:60]
                misc = (c.misconception or "")[:60]
                print(f"  cosine={cos:.4f}, chunk={c.chunk_id}: {topic} | {misc}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(run_with_logging(main, script_name="probe_retrieval"))
