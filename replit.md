# Company Match — Sana Commerce

## Overview
A Flask web application that lets users upload two company CSVs, configure matching parameters, and download a matched results file. Uses OpenAI embeddings for semantic matching with a fast exact-match pre-pass.

## Project Structure
- `app.py` — Flask web server (port 5000)
- `match_core.py` — Programmatic matching API (called by the web server)
- `match_companies.py` — Original CLI script (helpers imported by match_core.py)
- `templates/index.html` — Sana Commerce-branded frontend
- `requirements.txt` — Python dependencies
- `emb_cache.json` — Embedding cache (auto-generated, gitignored)

## Running the App
```bash
python app.py
```
Serves on http://0.0.0.0:5000

## How It Works
1. User uploads source + target CSVs via the web UI
2. POST /api/run starts a background matching job and returns a job_id
3. Frontend polls GET /api/status/<job_id> for progress log
4. On completion, result CSV auto-downloads via GET /api/download/<job_id>

## Matching Logic (Two Passes)
1. **Exact pass**: Canonicalize names (strip punctuation, legal suffixes, articles) → exact string match
2. **Semantic pass**: OpenAI embeddings + cosine similarity for unmatched companies

## Exposed Settings (UI)
- Threshold (default 0.82)
- Top candidates / topk (default 3)
- Embedding model (text-embedding-3-small / 3-large / ada-002)
- Source column name (default: company)
- Target column name (default: company)
- Source ID column (optional)
- Target ID column (default: account_id)
- OpenAI API key (overrides OPENAI_API_KEY env var)

## Environment Variables
- `OPENAI_API_KEY` — Required for semantic matching (can also be entered in the UI)

## Dependencies
- `flask` — Web server
- `openai>=1.0.0` — OpenAI API client
- `numpy>=1.20.0` — Vector math
- `Unidecode>=1.3.0` — Unicode/accent normalization

## Brand
Sana Commerce brand guidelines applied:
- Colors: Sana Heart #EB0F37, Sana Indigo #12123F, Sana Azure #5664F9, Sana Oatmeal #F4F3F0
- Gradient header: Indigo → Cobalt with red radial overlay
- Typography: Inter (PP Neue Montreal web fallback)
