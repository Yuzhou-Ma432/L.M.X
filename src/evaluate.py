"""Evaluate generated descriptions against the three input strategies.

For each generated description we record:
    * token usage, estimated cost, and wall-clock time (always available)
    * similarity to the dataset's official description, when one exists
      (TF-IDF cosine via scikit-learn, or a token-overlap fallback)

Writes `data/outputs/results.json` and a human-readable `data/outputs/summary.md`.
"""
from __future__ import annotations

import json
from pathlib import Path

import config

RESULTS_PATH = config.OUTPUT_DIR / "results.json"
SUMMARY_PATH = config.OUTPUT_DIR / "summary.md"


def _cost(usage: dict) -> float:
    return (usage.get("prompt_tokens", 0) / 1e6 * config.COST_PER_1M_INPUT
            + usage.get("completion_tokens", 0) / 1e6 * config.COST_PER_1M_OUTPUT)


def _similarity(a: str, b: str) -> float:
    """Cosine similarity between two short texts. TF-IDF when sklearn is
    available, otherwise a Jaccard overlap of tokens."""
    if not a or not b:
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer().fit_transform([a, b])
        return float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
    except Exception:  # noqa: BLE001 — graceful fallback if sklearn is absent
        ta, tb = set(a.lower().split()), set(b.lower().split())
        return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0


def load_generated() -> list[dict]:
    rows: list[dict] = []
    for strat_dir in config.GENERATED_DIR.iterdir():
        if not strat_dir.is_dir():
            continue
        for f in strat_dir.glob("*.json"):
            try:
                rows.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
    return rows


def evaluate(rows: list[dict] | None = None) -> list[dict]:
    rows = rows if rows is not None else load_generated()

    # Load official descriptions from profiles.
    desc_lookup: dict[str, str] = {}
    for p in config.PROFILE_DIR.glob("*.json"):
        try:
            prof = json.loads(p.read_text(encoding="utf-8"))
            desc_lookup[prof["id"]] = prof.get("description") or ""
        except Exception:  # noqa: BLE001
            continue

    results: list[dict] = []
    for r in rows:
        usage = r.get("usage", {})
        official = desc_lookup.get(r["id"], "")
        results.append({
            "id": r["id"],
            "name": r["name"],
            "strategy": r["strategy"],
            "description": r["description"],
            "model": r.get("model"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "elapsed_seconds": r.get("elapsed_seconds", 0.0),
            "estimated_cost_usd": round(_cost(usage), 6),
            "similarity_to_official": round(_similarity(r["description"], official), 4)
            if official else None,
            "has_official_desc": bool(official),
        })

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(results: list[dict]) -> str:
    from collections import defaultdict
    by_strat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_strat[r["strategy"]].append(r)

    lines = ["# Evaluation Summary", "",
             f"Total generations: {len(results)}", ""]
    for strat in ["baseline", "content", "semantic"]:
        rs = by_strat.get(strat, [])
        if not rs:
            continue
        sims = [r["similarity_to_official"] for r in rs if r["similarity_to_official"] is not None]
        lines += [
            f"## {strat}  (n={len(rs)})",
            f"- avg total tokens: {_avg([r['total_tokens'] for r in rs]):.1f}",
            f"- avg elapsed: {_avg([r['elapsed_seconds'] for r in rs]):.2f}s",
            f"- avg est. cost: ${_avg([r['estimated_cost_usd'] for r in rs]):.6f}",
            f"- avg similarity to official: {_avg(sims):.4f}  (n={len(sims)} with official desc)",
            "",
        ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)
