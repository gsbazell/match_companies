"""
Programmatic API for company matching — used by the Flask web app.
Wraps the logic from match_companies.py without CLI overhead.
"""

import csv
import os
import sys
import tempfile
from typing import Callable, Dict, List, Optional

import numpy as np

# Re-use all helpers from the CLI module
from match_companies import (
    canonicalize,
    company_aliases,
    make_queries,
    normalize_rows,
    read_csv,
    write_csv,
    topk_similar,
    EmbeddingCache,
    embed_texts,
    resolve_api_key,
)


def run_matching_core(
    source_path: str,
    target_path: str,
    output_path: str,
    source_col: str = "company",
    target_col: str = "company",
    source_id_col: str = "",
    target_id_col: str = "account_id",
    model: str = "text-embedding-3-small",
    threshold: float = 0.82,
    topk: int = 3,
    batch_size: int = 128,
    api_key: str = "",
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Run the two-pass company matching and write the output CSV.

    Returns a summary dict with match counts.
    """
    def log(msg: str):
        if log_fn:
            log_fn(msg)

    # --- Load and normalise ---
    log("Loading source list…")
    source_rows = normalize_rows(read_csv(source_path), source_col)
    log(f"  {len(source_rows)} source companies loaded.")

    log("Loading target list…")
    target_rows = normalize_rows(read_csv(target_path), target_col)
    log(f"  {len(target_rows)} target companies loaded.")

    # --- Pass 1: exact canonical ---
    log("Running exact canonical matching…")
    canon_to_target: Dict[str, List[Dict]] = {}
    for r in target_rows:
        canon_to_target.setdefault(r["_canon"], []).append(r)

    results: List[Dict] = []
    unresolved: List[int] = []

    for i, r in enumerate(source_rows):
        c = r["_canon"]
        exacts = canon_to_target.get(c, [])
        if c and exacts:
            best = exacts[0]
            results.append({
                "source_company": r["_raw_name"],
                "source_account_id": r.get(source_id_col, "") if source_id_col else "",
                "target_company": best["_raw_name"],
                "target_account_id": best.get(target_id_col, ""),
                "similarity": 1.0,
                "match_type": "exact_canonical",
                "alt_candidates": "; ".join(
                    (x["_raw_name"] + (f" [{x.get(target_id_col, '')}]" if x.get(target_id_col, "") else ""))
                    for x in exacts[1:]
                ) if len(exacts) > 1 else "",
            })
        else:
            unresolved.append(i)

    exact_count = len(results)
    log(f"  {exact_count} exact matches found. {len(unresolved)} remaining for semantic pass.")

    # --- Pass 2: embedding ---
    embedding_count = 0
    review_count = 0

    if unresolved:
        resolved_api_key = resolve_api_key(api_key)
        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key not found. Add OPENAI_API_KEY to the environment or enter it in Settings."
            )

        from openai import OpenAI
        client = OpenAI(api_key=resolved_api_key)
        cache = EmbeddingCache("emb_cache.json", model=model)

        log(f"Generating embeddings for {len(unresolved)} source + {len(target_rows)} target companies…")
        source_unres_names = [source_rows[i]["_raw_name"] for i in unresolved]
        target_names = [r["_raw_name"] for r in target_rows]

        source_queries = make_queries(source_unres_names)
        target_queries = make_queries(target_names)

        log("  Embedding source companies…")
        Q = embed_texts(source_queries, client, cache, batch_size=batch_size)
        log("  Embedding target companies…")
        X = embed_texts(target_queries, client, cache, batch_size=batch_size)

        log("  Computing similarity scores…")
        top_idx, top_sims = topk_similar(Q, X, k=topk)

        for row_i, (cand_idxs, cand_sims) in enumerate(zip(top_idx, top_sims)):
            source_idx = unresolved[row_i]
            source_row = source_rows[source_idx]
            added = False
            alts: List[str] = []

            for j, sim in zip(cand_idxs, cand_sims):
                target_row = target_rows[int(j)]
                if not added and sim >= threshold:
                    results.append({
                        "source_company": source_row["_raw_name"],
                        "source_account_id": source_row.get(source_id_col, "") if source_id_col else "",
                        "target_company": target_row["_raw_name"],
                        "target_account_id": target_row.get(target_id_col, ""),
                        "similarity": round(float(sim), 6),
                        "match_type": "embedding",
                        "alt_candidates": "",
                    })
                    added = True
                    embedding_count += 1
                else:
                    id_val = target_row.get(target_id_col, "")
                    id_sfx = f" [{id_val}]" if id_val else ""
                    alts.append(f"{target_row['_raw_name']}{id_sfx} ({sim:.4f})")

            if not added:
                top1 = target_rows[int(cand_idxs[0])]
                results.append({
                    "source_company": source_row["_raw_name"],
                    "source_account_id": source_row.get(source_id_col, "") if source_id_col else "",
                    "target_company": top1["_raw_name"],
                    "target_account_id": top1.get(target_id_col, ""),
                    "similarity": round(float(cand_sims[0]), 6),
                    "match_type": "review_needed",
                    "alt_candidates": "; ".join(alts[1:]) if len(alts) > 1 else "",
                })
                review_count += 1

    # --- Write output ---
    order = {"exact_canonical": 0, "embedding": 1, "review_needed": 2}
    results.sort(key=lambda r: (order.get(r["match_type"], 9), -r["similarity"]))

    fields = [
        "source_company", "source_account_id",
        "target_company", "target_account_id",
        "similarity", "match_type", "alt_candidates",
    ]
    write_csv(output_path, results, fields)

    summary = {
        "total": len(results),
        "exact": exact_count,
        "embedding": embedding_count,
        "review_needed": review_count,
    }
    log(f"Done. {len(results)} total rows written.")
    log(f"  Exact: {exact_count} | Embedding: {embedding_count} | Needs review: {review_count}")
    return summary
