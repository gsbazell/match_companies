# Company Matching Tool

Match tradeshow company names to CRM accounts using OpenAI embeddings and intelligent canonicalization.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Get your key from: https://platform.openai.com/api-keys
```

### 3. Run the Matching
```bash
python match_companies.py \
  --tradeshow tradeshow.csv \
  --crm crm.csv \
  --output matches.csv \
  --threshold 0.82 \
  --topk 3
```

## How It Works

The tool uses a two-pass matching approach:

1. **Fast Exact Pass**: Canonicalizes company names (removes punctuation, "Inc", "LLC", etc.) for exact matching
2. **Semantic Pass**: Uses OpenAI embeddings with cosine similarity for unmatched companies

## CSV Requirements

### Tradeshow CSV
Must have a column with company names (default column name: `company`)

Example:
```csv
company
Acme Corporation
XYZ Industries Inc.
```

### CRM CSV
Must have columns for company names and account IDs:
- Company names (default: `company`)
- Account IDs (default: `account_id`)

Example:
```csv
company,account_id
Acme Corp,ACC-001
XYZ Industries,ACC-002
```

**Note**: Column names can be customized with `--tradeshow-col`, `--crm-col`, and `--crm-id-col` flags.

## Output

The script generates a CSV with these columns:

- `tradeshow_company` - Original tradeshow company name
- `crm_candidate` - Matched CRM company name
- `crm_account_id` - CRM account ID
- `similarity` - Match confidence (0-1, or 1.0 for exact matches)
- `match_type` - One of:
  - `exact_canonical` - Exact match after canonicalization
  - `embedding` - Semantic match above threshold
  - `review_needed` - Best guess below threshold (needs manual review)
- `alt_candidates` - Other possible matches for review

## Configuration Options

```bash
# Required
--tradeshow FILE       Path to tradeshow CSV
--crm FILE            Path to CRM CSV

# Optional
--output FILE         Output CSV path (default: matches.csv)
--threshold FLOAT     Match threshold 0-1 (default: 0.82)
--topk INT           Top candidates to show (default: 3)
--model NAME         OpenAI model (default: text-embedding-3-small)
--tradeshow-col NAME  Tradeshow company column (default: company)
--crm-col NAME       CRM company column (default: company)
--crm-id-col NAME    CRM account ID column (default: account_id)
--cache FILE         Embedding cache path (default: emb_cache.json)
--batch-size INT     API batch size (default: 128)
```

## Tips

### Adjusting the Threshold
- Start with `--threshold 0.82` (default)
- Review the `review_needed` matches in your output
- If too many false positives: increase threshold (e.g., 0.85)
- If missing obvious matches: decrease threshold (e.g., 0.78)

### Cost Optimization
- The tool caches embeddings in `emb_cache.json`
- Rerunning with the same data is nearly free
- Using `text-embedding-3-small` balances cost and quality
- For higher quality: `--model text-embedding-3-large`

### Performance
- Default settings work well for most use cases
- For very large CRM lists (100k+ records), consider implementing FAISS/hnswlib indexing

## Example Workflow

```bash
# Initial run
python match_companies.py \
  --tradeshow event_companies.csv \
  --crm salesforce_accounts.csv \
  --output matched_accounts.csv \
  --threshold 0.82

# Review the output
# Check matches with match_type = "review_needed"

# Adjust threshold if needed and re-run
python match_companies.py \
  --tradeshow event_companies.csv \
  --crm salesforce_accounts.csv \
  --output matched_accounts_v2.csv \
  --threshold 0.85
```

## Troubleshooting

**"OpenAI API key not found"**
- Ensure `.env` file exists with `OPENAI_API_KEY=your_key`
- Or set environment variable: `export OPENAI_API_KEY=your_key`

**"Column 'company' not found"**
- Your CSV uses different column names
- Use `--tradeshow-col` or `--crm-col` to specify the correct columns

**Rate limit errors**
- The script automatically retries with backoff
- If persistent, reduce `--batch-size` (e.g., `--batch-size 64`)

## License

Internal tool - check with your organization's policies before external distribution.
