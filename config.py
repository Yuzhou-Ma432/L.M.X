"""Central configuration for the L.M.X project.

Reads secrets from a `.env` file (never committed) and exposes the constants
used by every stage of the pipeline: collect -> profile -> generate -> evaluate.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CATALOG_DIR = DATA_DIR / "catalog"
PROFILE_DIR = DATA_DIR / "profiles"
OUTPUT_DIR = DATA_DIR / "outputs"
GENERATED_DIR = OUTPUT_DIR / "generated"

for _d in (DATA_DIR, CATALOG_DIR, PROFILE_DIR, OUTPUT_DIR, GENERATED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- DeepSeek LLM ----------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# "deepseek-chat" is the stable alias; the API currently resolves it to the
# fast chat model (deepseek-flash). Use "deepseek-reasoner" for the reasoning
# model if you want to compare.
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# --- Data collection (NYC Open Data / Socrata) -----------------------------
SOCRATA_DOMAIN = "data.cityofnewyork.us"
SOCRATA_CATALOG_URL = f"https://{SOCRATA_DOMAIN}/api/catalog/v1"
SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN") or None  # optional, raises rate limits
NUM_DATASETS = int(os.getenv("NUM_DATASETS", "150"))

# --- Profiling -------------------------------------------------------------
PROFILE_ROWS = int(os.getenv("PROFILE_ROWS", "5000"))  # max rows fetched per dataset
SAMPLE_ROWS = int(os.getenv("SAMPLE_ROWS", "5"))       # sample values shown in prompts
TOP_VALUES_N = int(os.getenv("TOP_VALUES_N", "5"))     # top-N frequent values recorded
TOP_VALUES_MAX_CARDINALITY = int(os.getenv("TOP_VALUES_MAX_CARDINALITY", "30"))  # only for low-cardinality columns
MAX_VALUE_LEN = int(os.getenv("MAX_VALUE_LEN", "40"))   # max chars per sample/top value (bounds prompt size)

# --- Generation ------------------------------------------------------------
MAX_DESCRIPTION_TOKENS = int(os.getenv("MAX_DESCRIPTION_TOKENS", "200"))
SEMANTIC_MAX_TOKENS = int(os.getenv("SEMANTIC_MAX_TOKENS", "150"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# --- Cost estimates (USD per 1M tokens, DeepSeek list prices) --------------
# These are the author's best-known list prices and may drift over time.
# Override via .env if you have exact numbers. Prompt-cache discounts are
# ignored for simplicity.
COST_PER_1M_INPUT = float(os.getenv("COST_PER_1M_INPUT", "0.27"))
COST_PER_1M_OUTPUT = float(os.getenv("COST_PER_1M_OUTPUT", "1.10"))

# --- Evaluation: semantic similarity via embeddings ------------------------
# sentence-transformers model for embedding cosine similarity (vs TF-IDF).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
