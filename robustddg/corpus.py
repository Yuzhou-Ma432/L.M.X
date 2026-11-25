"""Corpus of portal datasets: ``<dir>/meta.json`` + ``<dir>/<id>.csv``.

meta.json: [{"id", "title", "description", "category"}]. ``description`` is
the portal's human-written description; it is used only to derive evaluation
queries, never shown to the generator.

Optional ``<dir>/qrels.csv`` (header ``query,relevance,doc_id``) supplies human
relevance judgements, e.g. converted from ECIR-DDG; otherwise one known-item
query per dataset is derived from its portal description.

`fetch_socrata` builds a corpus from a Socrata open-data portal (default NYC
Open Data). It deliberately mixes in real COVID-vaccine datasets so that the
attack's target queries have legitimate answers to displace.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from .search import tokenize

HEADERS = {"User-Agent": "robustddg-research-prototype/0.1"}


def load(corpus_dir: str | Path, max_rows: int = 2000) -> list[dict]:
    d = Path(corpus_dir)
    meta = json.loads((d / "meta.json").read_text())
    out = []
    for m in meta:
        f = d / f"{m['id']}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, nrows=max_rows, low_memory=False)
        df = df[[c for c in df.columns if not str(c).startswith(":@")]]  # Socrata internals
        out.append({**m, "df": df})
    return out


def load_qrels(corpus_dir: str | Path) -> dict[str, dict[str, float]] | None:
    f = Path(corpus_dir) / "qrels.csv"
    if not f.exists():
        return None
    q: dict[str, dict[str, float]] = {}
    with open(f, newline="") as fh:
        for row in csv.DictReader(fh):
            q.setdefault(row["query"], {})[row["doc_id"]] = float(row["relevance"])
    return q


def known_item_queries(datasets: list[dict], n_terms: int = 4) -> dict[str, str]:
    """doc_id -> query made of the most distinctive terms of the portal
    description, excluding title words (so the title alone can't answer it)."""
    docs = {d["id"]: tokenize(d.get("description") or d["title"]) for d in datasets}
    df = Counter(w for toks in docs.values() for w in set(toks))
    n = len(docs)
    out = {}
    for d in datasets:
        title = set(tokenize(d["title"]))
        tf = Counter(w for w in docs[d["id"]] if w not in title and not w.isdigit())
        scored = sorted(tf, key=lambda w: (-tf[w] * math.log(n / df[w]), w))
        terms = scored[:n_terms] or list(title)[:n_terms]
        out[d["id"]] = " ".join(terms)
    return out


# ------------------------------------------------------------------ fetching --

def fetch_socrata(out_dir: str | Path, domain: str = "data.cityofnewyork.us", n: int = 40,
                  rows: int = 2000, extra_queries: tuple[str, ...] = ("covid vaccine",),
                  n_extra: int = 3) -> None:
    import requests

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    api = "https://api.us.socrata.com/api/catalog/v1"

    def catalog(**params) -> list[dict]:
        r = requests.get(api, params={"domains": domain, "only": "dataset", **params},
                         headers=HEADERS, timeout=60)
        r.raise_for_status()
        return r.json()["results"]

    picked: dict[str, dict] = {}
    for q in extra_queries:
        for res in catalog(q=q, limit=n_extra * 3):
            if sum(1 for p in picked.values() if p.get("_q") == q) >= n_extra:
                break
            r = res["resource"]
            if len(r.get("description") or "") > 80:
                picked[r["id"]] = {**_meta(res), "_q": q}
    # Diverse remainder: most-viewed datasets, round-robin over categories.
    by_cat: dict[str, list[dict]] = {}
    for res in catalog(limit=400, order="page_views_total"):
        m = _meta(res)
        if len(m["description"]) > 80 and len(res["resource"].get("columns_name", [])) <= 40:
            by_cat.setdefault(m["category"], []).append(m)
    while len(picked) < n and any(by_cat.values()):
        for cat in list(by_cat):
            if by_cat[cat] and len(picked) < n:
                m = by_cat[cat].pop(0)
                picked.setdefault(m["id"], m)

    meta = []
    for m in picked.values():
        url = f"https://{domain}/resource/{m['id']}.csv"
        try:
            r = requests.get(url, params={"$limit": rows}, headers=HEADERS, timeout=120)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {m['id']}: {e}")
            continue
        if len(df) < 20 or df.shape[1] < 3:
            continue
        df.to_csv(out / f"{m['id']}.csv", index=False)
        meta.append({k: v for k, v in m.items() if not k.startswith("_")})
        print(f"  {m['id']}  {len(df):>5} rows  {m['title'][:60]}")
        time.sleep(0.2)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"wrote {len(meta)} datasets to {out}")


def _meta(res: dict) -> dict:
    r = res["resource"]
    desc = re.sub(r"\s+", " ", r.get("description") or "").strip()
    return {"id": r["id"], "title": r.get("name", "").strip(), "description": desc,
            "category": (res.get("classification") or {}).get("domain_category") or "Other"}
