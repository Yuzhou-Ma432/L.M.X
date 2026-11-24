# L.M.X — Automating Dataset Description Generation Using LLMs

LLM-powered dataset description generation for the Fall 2026 Big Data final
project. We profile ~150 tabular datasets from **NYC Open Data**, then use a
single LLM (DeepSeek) under three input strategies to write catalog
descriptions, and compare them on quality, time, and cost.

This project is a partial reproduction of
[AutoDDG](https://arxiv.org/abs/2502.01050), applied to NYC Open Data.

## The three strategies

| Strategy | What the LLM receives |
|----------|-----------------------|
| `baseline` | dataset title + column names only |
| `content`  | title + column names + dtypes + sample values + summary statistics |
| `semantic` | the `content` prompt, plus a first LLM pass that summarizes likely topic & uses |

The same system prompt, model, temperature, and output length are used for all
three so the comparison stays fair.

## Pipeline

```
main.py collect   ->  data/catalog/selected.json    (choose ~150 datasets)
main.py profile   ->  data/profiles/{id}.json       (pandas profiling)
main.py generate  ->  data/outputs/generated/{id}/{strategy}.json
main.py evaluate  ->  data/outputs/results.json + summary.md
```

`python main.py pipeline` runs all four stages end to end. Pass `--limit N` to
any stage (or `pipeline`) to process only the first `N` datasets — useful for a
smoke test before spending API money.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows / Git Bash
pip install -r requirements.txt

cp .env.example .env                 # then put your DeepSeek key in .env
```

`.env` is gitignored — never commit it. Optional: add a free
[Socrata app token](https://data.cityofnewyork.us/profile/edit/developer_settings)
to `.env` to raise the rate limit.

## Run

```bash
python main.py collect               # fetch catalog, select 150 datasets
python main.py pipeline --limit 5    # quick end-to-end on 5 datasets
python main.py evaluate              # aggregate cost / time / similarity
```

## Configuration

All knobs live in `config.py` and can be overridden via `.env`:

- `DEEPSEEK_MODEL` — default `deepseek-chat` (currently resolves to the fast
  chat model). Set `deepseek-reasoner` to compare.
- `NUM_DATASETS`, `PROFILE_ROWS`, `SAMPLE_ROWS`
- `MAX_DESCRIPTION_TOKENS`, `TEMPERATURE`
- `COST_PER_1M_INPUT` / `COST_PER_1M_OUTPUT` — list-price estimates; update if
  pricing changes.

## Evaluation notes

- **Cost / time / tokens** come straight from the API response (`usage`), so
  those numbers are exact.
- **Similarity to the official description** uses TF-IDF cosine (scikit-learn),
  or a token-overlap fallback. It is a coarse signal, not ground truth — many
  NYC Open Data descriptions are themselves incomplete. Manual review of a
  sample is still the main quality check, as the proposal states.
- The `semantic` strategy makes **two** LLM calls per dataset (a semantic
  summary, then the description), which the cost accounting already includes.

## Team

Fangyuan Lu (fl2840) · Yuzhou Ma (ym3725) · Steve Xu (sx2273)
