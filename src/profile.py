"""Profile each selected dataset with pandas.

For every dataset, download a bounded sample of rows through the SODA CSV
endpoint and compute the per-column facts the LLM will need: name, dtype,
missing-value rate, distinct count, summary statistics (numeric columns), and
a few example values. Writes one `data/profiles/{id}.json` per dataset.
"""
from __future__ import annotations

import json
import time
from io import StringIO

import pandas as pd
import requests

import config

HEADERS = {"User-Agent": "L.M.X-project/0.1 (student project)"}
if config.SOCRATA_APP_TOKEN:
    HEADERS["X-App-Token"] = config.SOCRATA_APP_TOKEN


def fetch_rows(sid: str, limit: int) -> pd.DataFrame:
    url = f"https://{config.SOCRATA_DOMAIN}/resource/{sid}.csv"
    resp = requests.get(url, params={"$limit": limit}, headers=HEADERS, timeout=180)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text), low_memory=False)


def _safe_num(v) -> float | None:
    try:
        if pd.isna(v):
            return None
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _describe_column(series: pd.Series) -> dict:
    nonnull = series.dropna()
    n_rows = len(series)
    info: dict = {
        "dtype": str(series.dtype),
        "missing": int(series.isna().sum()),
        "missing_pct": round(series.isna().sum() / n_rows * 100, 2) if n_rows else 0.0,
        "nunique": int(nonnull.nunique()),
        "sample_values": nonnull.astype(str).head(config.SAMPLE_ROWS).tolist(),
    }
    if pd.api.types.is_numeric_dtype(series) and len(nonnull):
        info["stats"] = {
            "min": _safe_num(nonnull.min()),
            "max": _safe_num(nonnull.max()),
            "mean": _safe_num(nonnull.mean()),
            "median": _safe_num(nonnull.median()),
        }
    return info


def profile_dataframe(df: pd.DataFrame) -> dict:
    # Drop Socrata-internal columns (e.g. ":@computed_region_...").
    df = df.loc[:, [c for c in df.columns if not c.startswith(":@")]]
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": [
            {"name": str(c), **_describe_column(df[c])}
            for c in df.columns
        ],
    }


def profile_one(meta: dict, force: bool = False) -> dict | None:
    sid = meta["id"]
    out_path = config.PROFILE_DIR / f"{sid}.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    try:
        df = fetch_rows(sid, config.PROFILE_ROWS)
    except Exception as e:  # noqa: BLE001 — a bad dataset must not stop the run
        print(f"  !! {sid} ({meta['name']}): fetch failed: {e}")
        return None

    profile = {
        "id": sid,
        "name": meta["name"],
        "description": meta["description"],
        "category": meta["category"],
        "permalink": meta["permalink"],
        "sampled_rows": config.PROFILE_ROWS,
        **profile_dataframe(df),
    }
    out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return profile


def profile_all(selected: list[dict], limit: int | None = None, force: bool = False) -> list[dict]:
    subset = selected[:limit] if limit else selected
    profiles: list[dict] = []
    for i, meta in enumerate(subset, 1):
        print(f"[{i}/{len(subset)}] profiling {meta['id']} — {meta['name'][:60]}")
        p = profile_one(meta, force=force)
        if p is not None:
            profiles.append(p)
        time.sleep(0.1)
    print(f"Profiled {len(profiles)}/{len(subset)} datasets.")
    return profiles
