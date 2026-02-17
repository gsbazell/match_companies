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
- `--source-id-col` and `--target-id-col` are optional metadata columns.
  If an ID column is absent, output `source_account_id` / `target_account_id`
  is left blank.

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


def load_dotenv(path: str = ".env") -> None:
    """Load key=value entries from a local .env file into os.environ.

    Existing environment values are preserved.
    """
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f"Warning: could not load {path}: {e}", file=sys.stderr)


def resolve_api_key(cli_api_key: str = "") -> str:
    """Resolve API key from CLI flag, env vars, or local .env."""
    if cli_api_key:
        return cli_api_key

    load_dotenv()
    return (os.getenv("OPENAI_API_KEY") or "").strip()

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
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    def as_valid_vec(raw) -> np.ndarray:
        try:
            v = np.array(raw, dtype=np.float32)
        except Exception:
            return None
        if v.ndim != 1 or v.size == 0:
            return None
        return v

    vecs: List[np.ndarray] = [None] * len(texts)
    to_query_idxs: List[int] = []
    expected_dim = None

    # Resolve from cache or queue for API
    for i, t in enumerate(texts):
        cached = cache.get(t)
        v = as_valid_vec(cached) if cached is not None else None
        if v is None:
            to_query_idxs.append(i)
            continue
        if expected_dim is None:
            expected_dim = int(v.size)
        if int(v.size) != expected_dim:
            # Cache entry is from a different shape; refresh from API.
            to_query_idxs.append(i)
            continue
        vecs[i] = v

    # Call API in batches for missing ones
    for chunk_idxs in batch(to_query_idxs, size=batch_size):
        chunk_inputs = [texts[i] for i in chunk_idxs]
        while True:
            try:
                resp = client.embeddings.create(
                    model=cache.model,
                    input=chunk_inputs
                )
                break
            except Exception as e:
                # Gentle backoff on transient issues/rate limits
                wait = 2.0
                print(f"Embedding error: {e}. Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)

        if len(resp.data) != len(chunk_idxs):
            raise RuntimeError(
                f"Embedding API returned {len(resp.data)} vectors for {len(chunk_idxs)} inputs."
            )

        # Place results back in their original positions
        for j, d in enumerate(resp.data):
            i = chunk_idxs[j]
            vec = as_valid_vec(d.embedding)
            if vec is None:
                raise RuntimeError(f"Invalid embedding payload for input index {i}.")
            if expected_dim is None:
                expected_dim = int(vec.size)
            if int(vec.size) != expected_dim:
                raise RuntimeError(
                    f"Inconsistent embedding dimension for input index {i}: got {vec.size}, expected {expected_dim}."
                )
            vecs[i] = vec
            cache.set(texts[i], d.embedding)

    missing = [i for i, v in enumerate(vecs) if v is None]
    if missing:
        first = missing[0]
        raise RuntimeError(
            f"Missing embedding vectors for {len(missing)} input(s); first missing index={first} text={texts[first]!r}."
        )

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

def run_matching(
    source_rows: List[Dict[str, str]],
    target_rows: List[Dict[str, str]],
    source_col: str = "company",
    target_col: str = "company",
    source_id_col: str = "",
    target_id_col: str = "account_id",
    threshold: float = 0.82,
    topk: int = 3,
    model: str = "text-embedding-3-small",
    cache_path: str = "emb_cache.json",
    batch_size: int = 128,
    api_key: str = "",
) -> List[Dict[str, str]]:
    """Core matching logic. Returns a list of match result dicts.

    Each result dict has keys: source_company, source_account_id,
    target_company, target_account_id, similarity, match_type, alt_candidates.
    """
    id_col = target_id_col

    # Normalize rows (adds _raw_name and _canon fields)
    source_rows = normalize_rows(source_rows, source_col)
    target_rows = normalize_rows(target_rows, target_col)

    # --- Pass 1: exact canonical matches ---
    canon_to_target: Dict[str, List[Dict[str, str]]] = {}
    for r in target_rows:
        canon_to_target.setdefault(r["_canon"], []).append(r)

    results: List[Dict[str, str]] = []
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
        resolved_key = api_key or resolve_api_key()
        if not resolved_key:
            raise SystemExit(
                "OpenAI API key not found. Set OPENAI_API_KEY, add it to a local .env file, "
                "or pass --api-key."
            )

        client = OpenAI(api_key=resolved_key)
        cache = EmbeddingCache(cache_path, model=model)

        source_unres_names = [source_rows[i]["_raw_name"] for i in unresolved]
        target_names = [r["_raw_name"] for r in target_rows]

        source_queries = make_queries(source_unres_names)
        target_queries = make_queries(target_names)

        Q = embed_texts(source_queries, client, cache, batch_size=batch_size)
        X = embed_texts(target_queries, client, cache, batch_size=batch_size)

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
                        "target_account_id": target_row.get(id_col, ""),
                        "similarity": round(float(sim), 6),
                        "match_type": "embedding",
                        "alt_candidates": "",
                    })
                    added = True
                else:
                    id_val = target_row.get(id_col, "")
                    id_sfx = f" [{id_val}]" if id_val else ""
                    alts.append(f"{target_row['_raw_name']}{id_sfx} ({sim:.4f})")

            if not added:
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

    # Sort for readability
    order = {"exact_canonical": 0, "embedding": 1, "review_needed": 2}
    results.sort(key=lambda r: (order.get(r["match_type"], 9), -r["similarity"]))
    return results


RESULT_FIELDS = ["source_company", "source_account_id", "target_company",
                 "target_account_id", "similarity", "match_type", "alt_candidates"]


def compute_stats(results: List[Dict[str, str]], threshold: float = 0.82) -> Dict:
    """Compute summary statistics from match results."""
    import collections
    type_counts = collections.Counter(r["match_type"] for r in results)
    emb_sims = [float(r["similarity"]) for r in results if r["match_type"] == "embedding"]
    rev_sims = [float(r["similarity"]) for r in results if r["match_type"] == "review_needed"]
    return {
        "total_matched": len(results),
        "by_type": dict(type_counts),
        "threshold": threshold,
        "embedding_similarity": {
            "mean": round(sum(emb_sims) / len(emb_sims), 4) if emb_sims else None,
            "min": round(min(emb_sims), 4) if emb_sims else None,
            "max": round(max(emb_sims), 4) if emb_sims else None,
        },
        "review_similarity": {
            "mean": round(sum(rev_sims) / len(rev_sims), 4) if rev_sims else None,
            "min": round(min(rev_sims), 4) if rev_sims else None,
            "max": round(max(rev_sims), 4) if rev_sims else None,
        },
    }


def main():
    """CLI entry point: parse args, run matching, write results."""
    p = argparse.ArgumentParser(description="Match companies from two lists using OpenAI embeddings.")
    p.add_argument("--source", required=True, help="CSV with source companies to match")
    p.add_argument("--target", required=True, help="CSV with target companies to match against")
    p.add_argument("--source-col", default="company", help="Column name in source CSV (default: company)")
    p.add_argument("--target-col", default="company", help="Column name in target CSV (default: company)")
    p.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model")
    p.add_argument("--output", default="matches.csv", help="Output CSV path")
    p.add_argument("--threshold", type=float, default=0.82, help="Cosine similarity threshold to accept a match")
    p.add_argument("--topk", type=int, default=3, help="How many candidates to show per source company")
    p.add_argument("--cache", default="emb_cache.json", help="Path to local embedding cache")
    p.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    p.add_argument("--api-key", default="", help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    p.add_argument("--target-id-col", default="account_id", help="Optional metadata ID column in target CSV (default: account_id); output is blank when absent")
    p.add_argument("--source-id-col", default="", help="Optional metadata ID column in source CSV; output is blank when absent")

    args = p.parse_args()
    api_key = resolve_api_key(args.api_key)

    source_rows = read_csv(args.source)
    target_rows = read_csv(args.target)

    results = run_matching(
        source_rows=source_rows,
        target_rows=target_rows,
        source_col=args.source_col,
        target_col=args.target_col,
        source_id_col=args.source_id_col,
        target_id_col=args.target_id_col,
        threshold=args.threshold,
        topk=args.topk,
        model=args.model,
        cache_path=args.cache,
        batch_size=args.batch_size,
        api_key=api_key,
    )

    write_csv(args.output, results, RESULT_FIELDS)

    print(f"Wrote {len(results)} rows to {args.output}")
    unmatched = sum(1 for r in results if r["match_type"] == "review_needed")
    print(f"{unmatched} need review (below threshold {args.threshold}).")

if __name__ == "__main__":
    main()
