#!/usr/bin/env python3
"""MCP server exposing company matching as agentic tools.

Tools:
  - inspect_csv: Examine a CSV file's structure (columns, row count, samples)
  - match_companies: Run the two-pass matching pipeline (exact + embedding)
  - analyze_results: Compute statistics from a completed match output CSV

Usage:
  # stdio transport (for Claude Code, Cursor, Codex, etc.)
  python mcp_server.py

  # Or via the mcp CLI
  mcp run mcp_server.py
"""

import csv
import json
import os
import sys

from mcp.server.fastmcp import FastMCP

# Add project directory to path so we can import match_companies
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import match_companies as mc

mcp = FastMCP(
    "company-matcher",
    instructions=(
        "You are a company matching assistant. Use inspect_csv to understand "
        "the user's data, match_companies to run the matching, and "
        "analyze_results to evaluate the output. Iterate on the threshold "
        "if too many results need review."
    ),
)


@mcp.tool()
def inspect_csv(file_path: str, sample_rows: int = 5) -> str:
    """Examine a CSV file and return its structure.

    Returns column names, total row count, and a sample of rows
    so you can determine which column contains company names and
    choose the right parameters for matching.

    Args:
        file_path: Path to the CSV file.
        sample_rows: Number of sample rows to include (default 5).
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": f"File not found: {file_path}"})

    rows = mc.read_csv(file_path)
    if not rows:
        return json.dumps({"error": "CSV is empty or has no data rows"})

    columns = list(rows[0].keys())
    samples = rows[:sample_rows]

    # Guess which column is the company name column
    company_col_candidates = []
    for col in columns:
        lower = col.lower().replace("_", " ").replace("-", " ")
        if any(kw in lower for kw in ["company", "name", "organization", "org", "account"]):
            company_col_candidates.append(col)

    return json.dumps({
        "file": file_path,
        "columns": columns,
        "row_count": len(rows),
        "sample_rows": samples,
        "likely_company_columns": company_col_candidates,
    }, indent=2)


@mcp.tool()
def match_companies(
    source_path: str,
    target_path: str,
    output_path: str = "matches.csv",
    source_col: str = "company",
    target_col: str = "company",
    source_id_col: str = "",
    target_id_col: str = "account_id",
    threshold: float = 0.82,
    topk: int = 3,
    model: str = "text-embedding-3-small",
) -> str:
    """Match company names from a source CSV against a target CSV.

    Uses a two-pass approach:
    1. Fast exact matching on canonicalized names (strips punctuation, legal
       suffixes like Inc/LLC/Corp, and articles).
    2. Semantic matching using OpenAI embeddings for remaining companies.

    Results are written to output_path and a summary is returned.

    Embeddings are cached locally so re-runs with the same data are fast.

    Args:
        source_path: Path to the source CSV (companies to match).
        target_path: Path to the target CSV (companies to match against).
        output_path: Where to write the results CSV (default: matches.csv).
        source_col: Column name containing company names in source CSV.
        target_col: Column name containing company names in target CSV.
        source_id_col: Optional ID column in source CSV (leave empty if none).
        target_id_col: Optional ID column in target CSV (default: account_id).
        threshold: Cosine similarity threshold to accept an embedding match (0.0-1.0).
        topk: Number of candidate matches to consider per source company.
        model: OpenAI embedding model to use.
    """
    for path, label in [(source_path, "Source"), (target_path, "Target")]:
        if not os.path.exists(path):
            return json.dumps({"error": f"{label} file not found: {path}"})

    source_rows = mc.read_csv(source_path)
    target_rows = mc.read_csv(target_path)

    if not source_rows:
        return json.dumps({"error": "Source CSV is empty"})
    if not target_rows:
        return json.dumps({"error": "Target CSV is empty"})

    # Validate column names exist
    if source_col not in source_rows[0]:
        return json.dumps({
            "error": f"Column '{source_col}' not found in source CSV",
            "available_columns": list(source_rows[0].keys()),
        })
    if target_col not in target_rows[0]:
        return json.dumps({
            "error": f"Column '{target_col}' not found in target CSV",
            "available_columns": list(target_rows[0].keys()),
        })

    results = mc.run_matching(
        source_rows=source_rows,
        target_rows=target_rows,
        source_col=source_col,
        target_col=target_col,
        source_id_col=source_id_col,
        target_id_col=target_id_col,
        threshold=threshold,
        topk=topk,
        model=model,
    )

    mc.write_csv(output_path, results, mc.RESULT_FIELDS)

    stats = mc.compute_stats(results, threshold=threshold)
    stats["source_count"] = len(source_rows)
    stats["target_count"] = len(target_rows)
    stats["output_file"] = output_path

    return json.dumps(stats, indent=2)


@mcp.tool()
def analyze_results(results_path: str, threshold: float = 0.82) -> str:
    """Analyze a completed match results CSV and return statistics.

    Use this after match_companies to understand the quality of matches,
    or to decide whether to adjust the threshold and re-run.

    Args:
        results_path: Path to the match results CSV (output of match_companies).
        threshold: The threshold that was used (for context in the stats).
    """
    if not os.path.exists(results_path):
        return json.dumps({"error": f"File not found: {results_path}"})

    with open(results_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return json.dumps({"error": "Results CSV is empty"})

    # Convert similarity strings to floats for stats
    for r in rows:
        try:
            r["similarity"] = float(r["similarity"])
        except (ValueError, KeyError):
            r["similarity"] = 0.0

    stats = mc.compute_stats(rows, threshold=threshold)

    # Add review_needed details for the agent to reason about
    review_rows = [r for r in rows if r["match_type"] == "review_needed"]
    review_sample = [
        {
            "source": r["source_company"],
            "matched_to": r["target_company"],
            "similarity": r["similarity"],
            "alternatives": r.get("alt_candidates", ""),
        }
        for r in review_rows[:10]
    ]

    # Near-threshold analysis: how many review_needed are close to threshold
    near_threshold = [r for r in review_rows
                      if r["similarity"] >= threshold - 0.05]

    stats["review_sample"] = review_sample
    stats["near_threshold_count"] = len(near_threshold)
    if near_threshold:
        stats["suggested_threshold"] = round(
            min(float(r["similarity"]) for r in near_threshold) - 0.01, 2
        )

    return json.dumps(stats, indent=2)


if __name__ == "__main__":
    mcp.run()
