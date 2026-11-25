"""Defenses against poisoned uploads (applied before/around the prompts).

    sanitize      strip instruction-like text and keyword-list cells from column
                  names and values (rename poisoned columns to ``column_<i>``)
    stats_only    treat free-text columns as untrusted: the LLM sees their name
                  and statistics only — no sample values, no frequent values
    consistency   cross-check (from prototype #1, applied to topics): derive the
                  topic from the full table and from the trusted view only; if
                  they share no content word, flag the dataset and generate
                  from the trusted view
"""
from __future__ import annotations

import re

import pandas as pd

_INSTR = re.compile(
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instruction|previous|above|prior|prompt)s?\b"
    r"|\b(describe|summari[sz]e|present|classify|label)\s+(this|the)\s+(dataset|data|table)\s+as\b"
    r"|\bnote\s+to\s+(the\s+)?(ai|llm|assistant|model)\b|\bas\s+an\s+ai\b|\bsystem\s*prompt\b"
    r"|\byou\s+(are|must|should)\b|<\|.*?\|>|\b(assistant|system|user)\s*:",
    re.I | re.S)


def is_instruction(s: object) -> bool:
    return isinstance(s, str) and bool(_INSTR.search(s))


def is_keyword_list(s: object) -> bool:
    """A cell that is a delimiter-separated bag of short phrases (SEO stuffing)."""
    if not isinstance(s, str):
        return False
    parts = [x.strip() for x in re.split(r"[;|]|,(?!\d)", s) if x.strip()]
    return len(parts) >= 5 and all(1 <= len(x.split()) <= 6 for x in parts)


def sanitize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    report = {"renamed_columns": [], "cleaned_cells": {}}
    new_cols = []
    for i, c in enumerate(out.columns):
        name = str(c)
        if is_instruction(name) or len(name) > 64:
            report["renamed_columns"].append(name)
            name = f"column_{i}"
        new_cols.append(name)
    out.columns = new_cols
    for c in out.columns:
        if out[c].dtype == object:
            bad = out[c].map(lambda v: is_instruction(v) or is_keyword_list(v))
            if bad.any():
                report["cleaned_cells"][c] = int(bad.sum())
                out.loc[bad, c] = None
    # A column emptied by cleaning carries nothing but attacker intent.
    emptied = [c for c in report["cleaned_cells"] if out[c].isna().all()]
    out = out.drop(columns=emptied)
    report["dropped_columns"] = emptied
    return out, report


def untrusted_columns(df: pd.DataFrame) -> set[str]:
    """Free-text-like string columns: long values or near-unique values."""
    out = set()
    for c in df.columns:
        s = df[c].dropna()
        if s.empty or s.dtype != object:
            continue
        st = s.astype(str)
        if st.str.len().max() > 40 or (s.nunique() / len(s) > 0.5 and st.str.len().mean() > 15):
            out.add(str(c))
    return out


def suspicious_names(df: pd.DataFrame) -> set[str]:
    return {str(c) for c in df.columns if is_instruction(str(c)) or len(str(c)) > 64}


_TOPIC_STOP = {"data", "dataset", "datasets", "and", "of", "the", "nyc", "new", "york", "city",
               "records", "information", "statistics", "by", "in", "for"}


def topic_words(t: str) -> set[str]:
    ws = re.findall(r"[a-z0-9]+", t.lower())
    return {w.rstrip("s") for w in ws if w not in _TOPIC_STOP and len(w) > 2}


def topics_agree(a: str, b: str) -> bool:
    wa, wb = topic_words(a), topic_words(b)
    return not wa or not wb or bool(wa & wb)
