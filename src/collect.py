"""Collect tabular datasets from NYC Open Data (Socrata).

Uses the Socrata "Data Discovery" (catalog v1) API to list datasets and select
a diverse, reproducible sample balanced across categories and description
richness. Writes:

    data/catalog/catalog.json   — raw-ish catalog dump (large, gitignored)
    data/catalog/selected.json  — the chosen datasets + metadata (committed)
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict

import requests

import config

HEADERS = {"User-Agent": "L.M.X-project/0.1 (student project; fl2840/ym3725/sx2273)"}
if config.SOCRATA_APP_TOKEN:
    HEADERS["X-App-Token"] = config.SOCRATA_APP_TOKEN


def fetch_catalog() -> list[dict]:
    """Page through the Socrata catalog and return every tabular dataset."""
    results: list[dict] = []
    limit, offset = 10000, 0
    while True:
        params = {
            "domains": config.SOCRATA_DOMAIN,
            "only": "dataset",
            "limit": limit,
            "offset": offset,
        }
        resp = requests.get(config.SOCRATA_CATALOG_URL, params=params,
                            headers=HEADERS, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("results", [])
        if not batch:
            break
        results.extend(batch)
        total = int(data.get("resultSetSize", 0))
        offset += len(batch)
        if offset >= total:
            break
        time.sleep(0.2)
    return results


def _normalize(ds: dict) -> dict | None:
    res = ds.get("resource") or {}
    if res.get("type") != "dataset":
        return None
    sid = res.get("id")
    if not sid:
        return None
    desc = (res.get("description") or "").strip()
    cls = ds.get("classification") or {}
    return {
        "id": sid,
        "name": (res.get("name") or "").strip(),
        "description": desc,
        "desc_words": len(desc.split()),
        "category": cls.get("domain_category") or "Uncategorized",
        "tags": cls.get("domain_tags") or [],
        "columns_name": res.get("columns_name") or [],
        "columns_datatype": res.get("columns_datatype") or [],
        "columns_description": res.get("columns_description") or [],
        "provenance": res.get("provenance") or "",
        "download_count": int(res.get("download_count") or 0),
        "permalink": ds.get("permalink") or "",
        "data_updated_at": res.get("data_updated_at") or res.get("updatedAt") or "",
    }


def _richness_bucket(words: int) -> str:
    """Coarse buckets for how detailed an existing description is."""
    if words == 0:
        return "none"
    if words <= 30:
        return "short"
    if words <= 120:
        return "medium"
    return "long"


def select_stratified(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Deterministic stratified sample: round-robin across richness buckets,
    with category diversity inside each bucket."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        buckets[_richness_bucket(r["desc_words"])].append(r)

    # Sort each bucket by category so adjacent round-robin picks differ.
    for b in buckets.values():
        b.sort(key=lambda r: (r["category"], -r["download_count"]))

    order = ["none", "short", "medium", "long"]
    idx = {k: 0 for k in order}
    selected: list[dict] = []
    while len(selected) < n:
        progressed = False
        for k in order:
            lst = buckets.get(k, [])
            if idx[k] < len(lst):
                selected.append(lst[idx[k]])
                idx[k] += 1
                progressed = True
                if len(selected) >= n:
                    break
        if not progressed:
            break

    # Top up from any remaining records if we ran short of a bucket.
    if len(selected) < n:
        rest = [r for k in order for r in buckets.get(k, [])[idx[k]:]]
        rng = random.Random(seed)
        rng.shuffle(rest)
        seen = {r["id"] for r in selected}
        for r in rest:
            if r["id"] in seen:
                continue
            selected.append(r)
            if len(selected) >= n:
                break
    return selected


def collect(limit: int | None = None) -> list[dict]:
    n = limit if limit is not None else config.NUM_DATASETS
    print(f"Fetching catalog from {config.SOCRATA_DOMAIN} ...")
    raw = fetch_catalog()
    records = [r for r in (_normalize(d) for d in raw) if r is not None]
    print(f"  {len(raw)} catalog entries -> {len(records)} usable tabular datasets")

    (config.CATALOG_DIR / "catalog.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    selected = select_stratified(records, n)
    (config.CATALOG_DIR / "selected.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Quick summary of what we picked.
    from collections import Counter
    print(f"Selected {len(selected)} datasets:")
    print("  by category:", dict(Counter(r['category'] for r in selected).most_common(8)))
    print("  by desc richness:", dict(Counter(_richness_bucket(r['desc_words']) for r in selected)))
    return selected
