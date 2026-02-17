# Company Matching Tool - Project Context

## Overview
This project matches company names from two lists using OpenAI embeddings and intelligent canonicalization techniques. It's a generic tool that can compare any two lists of companies (e.g., tradeshow attendees vs CRM records, vendor lists vs master databases, etc.).

## Project Structure
- `match_companies.py` - Main matching script (also importable as a library)
- `mcp_server.py` - MCP server exposing matching as agentic tools
- `.mcp.json` - MCP server registration for Claude Code / compatible agents
- `requirements.txt` - Python dependencies
- `emb_cache.json` - Embedding cache (auto-generated, should be in .gitignore)
- Input CSVs: source companies and target companies
- Output: CSV with matches, similarities, and match types

## How It Works
The script uses a two-pass approach:
1. **Fast exact pass**: Canonicalizes names (removes punctuation, legal suffixes, articles) for exact string matching
2. **Semantic pass**: Uses OpenAI embeddings with cosine similarity for unmatched companies

## Key Features
- Generates company aliases (acronyms, collapsed names) to improve matching
- Caches embeddings locally to avoid repeated API calls
- Outputs match confidence scores and alternative candidates
- Three match types: `exact_canonical`, `embedding`, `review_needed`

## Dependencies
Core CLI dependencies (`requirements.txt`):
- `openai` - For embeddings API
- `numpy` - For vector operations
- `Unidecode` - Optional, for accent folding in company names

MCP server adds (`requirements-mcp.txt`):
- `mcp` - Model Context Protocol SDK
- `openpyxl` - Excel file reading

## Environment Variables
- `OPENAI_API_KEY` - Required for OpenAI API access

## Usage Pattern
```bash
python match_companies.py \
  --source list1.csv \
  --target list2.csv \
  --output matches.csv \
  --threshold 0.82 \
  --topk 3
```

## CSV Format Requirements
Both input CSVs need:
- Source CSV: column with company names (default: `company`), optional ID column
- Target CSV: columns for company names (default: `company`) and account IDs (default: `account_id`)

Override column names with `--source-col`, `--target-col`, `--source-id-col`, and `--target-id-col` flags.

## Code Style Notes
- Uses minimal dependencies intentionally
- Prefers standard library where possible
- Graceful fallback if Unidecode not available
- Simple JSON-based caching for embeddings
- UTF-8-sig encoding for Excel compatibility

## Performance Considerations
- Default embedding model: `text-embedding-3-small` (balance of speed/cost)
- Batch size: 128 texts per API call
- For very large target lists (100k+), consider replacing `topk_similar` with FAISS/hnswlib ANN index

## MCP Server (Agentic Skill)

The project includes an MCP server (`mcp_server.py`) that exposes company matching as tools for AI agents. This works with Claude Code, Cursor, Codex, and any MCP-compatible agent.

### Tools Provided
- **`inspect_file`** - Examine a CSV or Excel file's structure (sheets, columns, row count, sample data, guessed company/ID columns)
- **`prepare_csv`** - Extract company name and ID columns from a messy Excel or CSV into a clean matching-ready CSV (handles dedup, blank rows, column renaming)
- **`match_companies`** - Run the full two-pass matching pipeline and return results + statistics
- **`analyze_results`** - Compute statistics from a completed match output, including threshold tuning suggestions

### Registration
The server is registered in `.mcp.json` at the project root. For Claude Code, it auto-discovers on session start. For other agents, point to `mcp_server.py` as a stdio MCP server.

### Agentic Workflow
An agent using these tools should:
1. Call `inspect_file` on both input files to understand structure (columns, sheets for Excel, likely company/ID columns)
2. Call `prepare_csv` to extract the relevant columns into clean CSVs with standardized names
3. Call `match_companies` with the prepared CSVs
4. Call `analyze_results` to evaluate match quality
5. If too many `review_needed`, adjust threshold and re-run (embeddings are cached, so re-runs are fast)
