#!/usr/bin/env python

"""
Match companies from two lists using OpenAI embeddings.

This script does two passes:
  1) **Fast exact pass** on a *canonicalized* version of names
     (e.g., drops punctuation, legal suffixes like "Inc."/"LLC", leading
     articles like "the"). If a canonical string matches exactly, we accept it.
  2) **Semantic pass** using embeddings for anything that wasn't matched in (1).
     We generate a few short aliases ("acme corp" → "ac", "acmecorp"), embed
     those alias strings, and rank target candidates by cosine similarity.

Output: a CSV with columns: source_company, source_account_id, target_company,
target_account_id, similarity, match_type (exact_canonical | embedding | review_needed),
and alt_candidates.

Good defaults:
- Use `text-embedding-3-small` for speed/cost. Change with `--model`.
- Start with `--threshold 0.82` and adjust after sampling a few results.

Typical usage (one-liner):
    python match_companies.py --source list1.csv --target list2.csv --output matches.csv --threshold 0.82 --topk 3

CSV expectations:
- Both files should have a column with the company name (default column name:
  `company`). You can override with `--source-col` and `--target-col`.
  The target CSV should also have a column with the account id (default column name:
  'account_id')

Notes:
- The script caches embeddings in `emb_cache.json` so you only pay for new rows.
- Cosine similarity is implemented as a dot product because we L2-normalize
  vectors.
- If you have very large target lists (hundreds of thousands), consider swapping
  `topk_similar` for an ANN index (e.g., FAISS or hnswlib) — the rest stays the
  same.
"""


import argparse, csv, hashlib, json, math, os, re, sys, time
from typing import List, Dict, Tuple
import numpy as np

# Optional but nice: pip install Unidecode for accent folding; otherwise we fall back gracefully
try:
    from unidecode import unidecode
except Exception:
    unidecode = None

# OpenAI SDK (pip install openai)
from openai import OpenAI

LEGAL_SUFFIXES = {
    "inc", "inc.", "llc", "l.l.c.", "ltd", "ltd.", "co", "co.",
    "corp", "corp.", "company", "gmbh", "s.a.", "s.a", "sarl", "corporation",
    "bv", "pte", "pte.", "plc", "plc.", "ag", "oy", "aps", "as",
    "ab", "kk", "k.k.", "oyj", "nv", "s.p.a.", "spa", "oy", "srl", "s.r.l.",
}
ARTICLES = {"the", "a", "an"}

def strip_legal_suffixes(tokens: List[str]) -> List[str]:
    # Remove trailing legal suffix tokens
    out = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return out

_punct = re.compile(r"[^\w\s]")
_ws = re.compile(r"\s+")
def canonicalize(name: str) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    if unidecode:
        s = unidecode(s)
    s = _punct.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    toks = s.split()
    if toks and toks[0] in ARTICLES:
        toks = toks[1:]
    toks = strip_legal_suffixes(toks)
    return " ".join(toks)

def company_aliases(name: str) -> List[str]:
    """Generate a few cheap aliases to help the embedding."""
    c = canonicalize(name)
    if not c:
        return []
    toks = c.split()
    acronym = "".join(t[0] for t in toks if t)
    aliases = [c]
    if len(acronym) >= 2:
        aliases.append(acronym)
    # hyphen/collapse variants
    aliases.append(c.replace(" ", ""))
    return list(dict.fromkeys(aliases))  # unique, keep order

def batch(iterable, size=128):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf

def normalize_rows(rows: List[Dict[str, str]], col: str) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        raw = (r.get(col) or "").strip()
        out.append({
            **r,
            "_raw_name": raw,
            "_canon": canonicalize(raw),
        })
    return out

def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]):
    # Use utf-8-sig so Excel opens the CSV with correct UTF-8 decoding
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

class EmbeddingCache:
    """Simple JSON cache on disk keyed by (model, text hash)."""
    def __init__(self, path="emb_cache.json", model="text-embedding-3-small"):
        self.path = path
        self.model = model
        self.data = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def _key(self, text: str) -> str:
        h = hashlib.sha256((self.model + "||" + text).encode("utf-8")).hexdigest()
        return h

    def get(self, text: str):
        return self.data.get(self._key(text))

    def set(self, text: str, vec):
        self.data[self._key(text)] = vec

    def flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)

def embed_texts(texts: List[str], client: OpenAI, cache: EmbeddingCache, batch_size=128) -> np.ndarray:
    vecs = []
    to_query = []
    idx_map = {}

    # Resolve from cache or queue for API
    for i, t in enumerate(texts):
        cached = cache.get(t)
        if cached is not None:
            vecs.append(np.array(cached, dtype=np.float32))
        else:
            idx_map[len(to_query)] = i
            to_query.append(t)
            vecs.append(None)  # placeholder

    # Call API in batches for missing ones
    for chunk in batch(to_query, size=batch_size):
        while True:
            try:
                resp = client.embeddings.create(
                    model=cache.model,
                    input=chunk
                )
                break
            except Exception as e:
                # Gentle backoff on transient issues/rate limits
                wait = 2.0
                print(f"Embedding error: {e}. Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)

        # Place results back in their positions
        for j, d in enumerate(resp.data):
            i = idx_map[j]
            vec = np.array(d.embedding, dtype=np.float32)
            vecs[i] = vec
            cache.set(texts[i], d.embedding)

    cache.flush()
    # Stack and L2-normalize for cosine as dot product
    M = np.vstack(vecs)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return M / norms

def make_queries(names: List[str]) -> List[str]:
    # For each name, create a short joined alias string to improve signal
    # e.g., "acme corp" + "ac" + "acmecorp"
    out = []
    for n in names:
        aliases = company_aliases(n)
        out.append(" | ".join(aliases))
    return out

def topk_similar(
    Q: np.ndarray,  # queries (Nq x d)
    X: np.ndarray,  # index   (Nx x d)
    k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    # cosine similarity = dot product since already normalized
    sims = Q @ X.T  # (Nq, Nx)
    # top-k via argpartition for speed
    k = min(k, X.shape[0])
    idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
    # sort those k
    part_sims = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-part_sims, axis=1)
    top_idx = np.take_along_axis(idx, order, axis=1)
    top_sims = np.take_along_axis(part_sims, order, axis=1)
    return top_idx, top_sims

def main():
    """CLI entry point: parse args, run matching, write results."""
    p = argparse.ArgumentParser(description="Match companies from two lists using OpenAI embeddings.")
    # Required paths for the two CSVs
    p.add_argument("--source", required=True, help="CSV with source companies to match")
    p.add_argument("--target", required=True, help="CSV with target companies to match against")

    # Column names if they aren't the default `company`
    p.add_argument("--source-col", default="company", help="Column name in source CSV (default: company)")
    p.add_argument("--target-col", default="company", help="Column name in target CSV (default: company)")

    # Embedding model + knobs
    p.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model")
    p.add_argument("--output", default="matches.csv", help="Output CSV path")
    p.add_argument("--threshold", type=float, default=0.82, help="Cosine similarity threshold to accept a match")
    p.add_argument("--topk", type=int, default=3, help="How many candidates to show per source company")
    p.add_argument("--cache", default="emb_cache.json", help="Path to local embedding cache")
    p.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")

    # Target account ID column
    p.add_argument("--target-id-col", default="account_id", help="Column in target CSV with the unique account ID (default: account_id)")

    # Source account ID column (optional)
    p.add_argument("--source-id-col", default="", help="Column in source CSV with unique source ID (optional)")

    args = p.parse_args()
    id_col = args.target_id_col
    source_id_col = args.source_id_col

    # --- Load and normalize input data ---
    source_rows = normalize_rows(read_csv(args.source), args.source_col)
    target_rows = normalize_rows(read_csv(args.target), args.target_col)

    # --- Pass 1: exact canonical matches ---
    # Build an index from canonical name → list of target rows that share it.
    canon_to_target: Dict[str, List[Dict[str, str]]] = {}
    for r in target_rows:
        canon_to_target.setdefault(r["_canon"], []).append(r)

    results: List[Dict[str, str]] = []
    unresolved: List[int] = []  # indexes of source rows to process in the embedding pass

    for i, r in enumerate(source_rows):
        c = r["_canon"]
        exacts = canon_to_target.get(c, [])
        if c and exacts:
            # If multiple target rows share the same canonical, accept the first and
            # record the rest as alternatives for human review.
            best = exacts[0]
            results.append({
                "source_company": r["_raw_name"],
                "source_account_id": r.get(source_id_col, "") if source_id_col else "",
                "target_company": best["_raw_name"],
                "target_account_id": best.get(id_col, ""),
                "similarity": 1.0,
                "match_type": "exact_canonical",
                "alt_candidates": "; ".join(
                    (f"{x['_raw_name']}" + (f" [{x.get(id_col, '')}]" if x.get(id_col, '') else ""))
                    for x in exacts[1:]
                ) if len(exacts) > 1 else "",
            })
        else:
            unresolved.append(i)

    # --- Pass 2: embedding-based matches for unresolved rows ---
    if unresolved:
        client = OpenAI()  # uses OPENAI_API_KEY from your environment
        cache = EmbeddingCache(args.cache, model=args.model)

        # Collect display names (original strings) for unresolved source rows
        source_unres_names = [source_rows[i]["_raw_name"] for i in unresolved]
        target_names = [r["_raw_name"] for r in target_rows]

        # Convert names → alias strings → embeddings
        source_queries = make_queries(source_unres_names)
        target_queries = make_queries(target_names)

        Q = embed_texts(source_queries, client, cache, batch_size=args.batch_size)  # Nq x d
        X = embed_texts(target_queries, client, cache, batch_size=args.batch_size)  # Nx x d

        # For each source query, retrieve the top-k target candidates + scores
        top_idx, top_sims = topk_similar(Q, X, k=args.topk)

        # Walk each row's candidates and apply the acceptance threshold
        for row_i, (cand_idxs, cand_sims) in enumerate(zip(top_idx, top_sims)):
            source_idx = unresolved[row_i]
            source_row = source_rows[source_idx]
            added = False  # whether we've accepted a candidate above threshold
            alts: List[str] = []

            for j, sim in zip(cand_idxs, cand_sims):
                target_row = target_rows[int(j)]
                if not added and sim >= args.threshold:
                    # Accept the first candidate that clears the threshold
                    results.append({
                        "source_company": source_row["_raw_name"],
                        "source_account_id": source_row.get(source_id_col, "") if source_id_col else "",
                        "target_company": target_row["_raw_name"],
                        "target_account_id": target_row.get(id_col, ""),
                        "similarity": round(float(sim), 6),
                        "match_type": "embedding",
                        "alt_candidates": "",
                    })
                    added = True
                else:
                    # Keep other candidates (and sub-threshold ones) as alternates
                    id_val = target_row.get(id_col, "")
                    id_sfx = f" [{id_val}]" if id_val else ""
                    alts.append(f"{target_row['_raw_name']}{id_sfx} ({sim:.4f})")

            if not added:
                # No candidate cleared the threshold. We still record the top-1
                # so you have something to review manually.
                top1 = target_rows[int(cand_idxs[0])]
                results.append({
                    "source_company": source_row["_raw_name"],
                    "source_account_id": source_row.get(source_id_col, "") if source_id_col else "",
                    "target_company": top1["_raw_name"],
                    "target_account_id": top1.get(id_col, ""),
                    "similarity": round(float(cand_sims[0]), 6),
                    "match_type": "review_needed",
                    "alt_candidates": "; ".join(alts[1:]) if len(alts) > 1 else "",
                })

    # --- Output: sort for readability and write CSV ---
    order = {"exact_canonical": 0, "embedding": 1, "review_needed": 2}
    results.sort(key=lambda r: (order.get(r["match_type"], 9), -r["similarity"]))

    fields = ["source_company", "source_account_id", "target_company", "target_account_id", "similarity", "match_type", "alt_candidates"]
    write_csv(args.output, results, fields)

    # Log a quick summary to stdout
    print(f"Wrote {len(results)} rows to {args.output}")
    unmatched = sum(1 for r in results if r["match_type"] == "review_needed")
    print(f"{unmatched} need review (below threshold {args.threshold}).")

if __name__ == "__main__":
    main()
