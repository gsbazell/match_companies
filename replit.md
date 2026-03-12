# Company Matching Tool

## Overview
A Python command-line tool that matches company names from two CSV lists using OpenAI embeddings and intelligent canonicalization. Useful for matching event attendees vs CRM records, vendor lists vs databases, etc.

## Project Structure
- `match_companies.py` - Main CLI script
- `requirements.txt` - Python dependencies (openai, numpy, Unidecode)
- `.env.example` - Template for environment variables
- `emb_cache.json` - Embedding cache (auto-generated, gitignored)

## How to Run
```bash
python match_companies.py \
  --source list1.csv \
  --target list2.csv \
  --output matches.csv \
  --threshold 0.82 \
  --topk 3
```

## Environment Variables
- `OPENAI_API_KEY` - Required for OpenAI embeddings API

## Dependencies
- `openai>=1.0.0` - OpenAI API client
- `numpy>=1.20.0` - Vector operations
- `Unidecode>=1.3.0` - Accent/unicode normalization

## Architecture
Two-pass matching approach:
1. **Exact pass**: Canonicalizes names (removes punctuation, legal suffixes like Inc/LLC) for fast exact matching
2. **Semantic pass**: Uses OpenAI `text-embedding-3-small` embeddings with cosine similarity for fuzzy matching

Output columns: `source_company`, `source_account_id`, `target_company`, `target_account_id`, `similarity`, `match_type`, `alt_candidates`

Match types: `exact_canonical`, `embedding`, `review_needed`
