"""Statistical profiler (a light stand-in for datamart-profiler used by AutoDDG).

A profile is a plain dict so that `inject_errors` can corrupt it surgically:

    {"title", "n_rows", "n_cols",
     "columns": [{"name", "type", "n_distinct", "missing_ratio",
                  "min", "max", "mean", "std",           # numeric
                  "date_min", "date_max",                 # datetime
                  "unit", "top_values", "samples"}]}

``samples`` are raw cell strings drawn from the data and are what the semantic
profiler sees; every other field is a *derived statistic*. A profiler bug
corrupts the derived fields but not the raw samples — that asymmetry is what
the cross-check in `crosscheck.py` exploits.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

SAMPLE_SEED = 9      # AutoDDG uses random_state=9 for its 5-row sample
N_SAMPLES = 5
MAX_VALUE_LEN = 60


def sample_rows(df: pd.DataFrame, n: int = N_SAMPLES) -> pd.DataFrame:
    return df.sample(n, random_state=SAMPLE_SEED) if len(df) > n else df


def _clip(v: object) -> str:
    s = "" if pd.isna(v) else str(v)
    return s if len(s) <= MAX_VALUE_LEN else s[: MAX_VALUE_LEN - 1] + "…"


_DATEISH = re.compile(r"^\d{4}-\d{2}-\d{2}|^\d{1,2}/\d{1,2}/\d{2,4}")

_UNIT_RULES = [
    (r"(^|_)(pct|percent|percentage|rate_pct)($|_)|%", "percent"),
    (r"usd|dollar|amount|cost|price|salary|fee|budget|revenue|spend|paid", "US dollars"),
    (r"sq_?ft|square_?feet", "square feet"),
    (r"(^|_)acres?($|_)", "acres"),
    (r"miles?($|_)|mileage", "miles"),
    (r"(^|_)(lat|latitude)($|_)", "degrees latitude"),
    (r"(^|_)(lon|lng|longitude)($|_)", "degrees longitude"),
    (r"(^|_)age($|_)", "years of age"),
    (r"temp|temperature", "degrees Fahrenheit"),
    (r"count|number_of|(^|_)num_|(^|_)n_", "count"),
]


def infer_unit(name: str, values: pd.Series | None = None) -> str | None:
    n = name.lower().replace(" ", "_")
    for pat, unit in _UNIT_RULES:
        if re.search(pat, n):
            return unit
    if values is not None and values.astype(str).str.startswith("$").mean() > 0.5:
        return "US dollars"
    return None


def _infer_type(s: pd.Series) -> tuple[str, pd.Series]:
    nonnull = s.dropna()
    if nonnull.empty:
        return "empty", s
    if pd.api.types.is_bool_dtype(s):
        return "boolean", s
    if pd.api.types.is_numeric_dtype(s):
        return ("integer" if np.allclose(nonnull % 1, 0) else "float"), s
    as_str = nonnull.astype(str).str.strip()
    num = pd.to_numeric(as_str.str.replace(r"^\$|,", "", regex=True), errors="coerce")
    if num.notna().mean() > 0.95:
        conv = pd.to_numeric(s.astype(str).str.replace(r"^\$|,", "", regex=True), errors="coerce")
        return ("integer" if np.allclose(num.dropna() % 1, 0) else "float"), conv
    if as_str.str.match(_DATEISH).mean() > 0.9:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt = pd.to_datetime(s, errors="coerce")
        if dt.notna().mean() > 0.9 * (len(nonnull) / max(len(s), 1)):
            return "datetime", dt
    if as_str.str.lower().isin({"true", "false", "yes", "no", "y", "n"}).mean() > 0.95:
        return "boolean", s
    avg_len = as_str.str.len().mean()
    if nonnull.nunique() <= max(30, 0.05 * len(nonnull)) and avg_len < 40:
        return "categorical", s
    return "text", s


def _r(x: float) -> float:
    return float(f"{x:.4g}")


def profile_df(df: pd.DataFrame, title: str) -> dict:
    samp = sample_rows(df)
    cols = []
    for name in df.columns:
        s = df[name]
        typ, conv = _infer_type(s)
        c = {"name": str(name), "type": typ,
             "n_distinct": int(s.nunique(dropna=True)),
             "missing_ratio": round(float(s.isna().mean()), 4),
             "unit": None,
             "samples": [_clip(v) for v in samp[name].tolist()],
             "top_values": []}
        if typ in ("integer", "float"):
            v = conv.dropna().astype(float)
            c.update(min=_r(v.min()), max=_r(v.max()), mean=_r(v.mean()), median=_r(v.median()),
                     std=_r(v.std()) if len(v) > 1 else 0.0,
                     unit=infer_unit(str(name), s))
        elif typ == "datetime":
            v = conv.dropna()
            c.update(date_min=v.min().strftime("%Y-%m-%d"), date_max=v.max().strftime("%Y-%m-%d"))
        if typ in ("categorical", "boolean"):
            vc = s.astype(str).value_counts().head(5)
            c["top_values"] = [_clip(k) for k in vc.index]
        cols.append(c)
    return {"title": title, "n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": cols}


# ---------------------------------------------------------------------------
# Rendering (AutoDDG `summarize` style) — `withheld` lets the cross-check hide
# or annotate disputed facts per column.
# ---------------------------------------------------------------------------

def render_column(c: dict, withheld: dict[str, str] | None = None) -> str:
    """withheld: {field_group: reason} where field_group in {range, unit, type}."""
    withheld = withheld or {}
    out = f"**{c['name']}**: "
    if "type" in withheld:
        out += (f"Data type uncertain ({withheld['type']}); raw sampled values: "
                f"{', '.join(s for s in c.get('samples', []) if s)}. ")
    else:
        out += f"Data is of type {c['type']}. "
    out += f"There are {c['n_distinct']} unique values. "
    out += f"Missing values: {c['missing_ratio']:.1%}. "
    if c["type"] in ("integer", "float") and "min" in c and "type" not in withheld:
        if "range" in withheld:
            out += (f"Value range uncertain ({withheld['range']}); raw sampled values: "
                    f"{', '.join(s for s in c.get('samples', []) if s)}. ")
        else:
            out += f"Mean {c['mean']:g}, standard deviation {c['std']:g}. "
            out += f"Minimum {c['min']:g}, maximum {c['max']:g}. "
    if c["type"] == "datetime" and "date_min" in c and "type" not in withheld:
        if "range" in withheld:
            out += (f"Date range uncertain ({withheld['range']}); raw sampled values: "
                    f"{', '.join(s for s in c.get('samples', []) if s)}. ")
        else:
            out += f"Date range: {c['date_min']} to {c['date_max']}. "
    if c.get("unit"):
        if "unit" in withheld:
            out += f"Unit uncertain ({withheld['unit']}). "
        else:
            out += f"Unit: {c['unit']}. "
    if c.get("top_values"):
        out += f"Most frequent values: {', '.join(c['top_values'])}. "
    return out.strip()


def render_profile(p: dict, withheld: dict[str, dict[str, str]] | None = None,
                   hide_columns: set[str] | None = None) -> str:
    withheld = withheld or {}
    hide_columns = hide_columns or set()
    lines = [f"Rows: {p['n_rows']}. Columns: {p['n_cols']}."]
    for c in p["columns"]:
        if c["name"] in hide_columns:
            continue
        lines.append(render_column(c, withheld.get(c["name"])))
    return "The key data profile information for this dataset includes:\n" + "\n".join(lines)
