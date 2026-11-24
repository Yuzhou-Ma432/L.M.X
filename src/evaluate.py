"""Evaluate generated descriptions against the three input strategies.

For each generated description we record:
    * token usage, estimated cost, and wall-clock time (always available)
    * similarity to the dataset's official description, when one exists,
      measured two ways:
        - TF-IDF cosine (lexical overlap, scikit-learn)
        - sentence-transformer embedding cosine (semantic similarity)

Writes `data/outputs/results.json` and `data/outputs/summary.md`, and reports
the per-strategy means of both metrics plus their correlation, so we can
justify which metric to report.
"""
from __future__ import annotations

import json

import config

RESULTS_PATH = config.OUTPUT_DIR / "results.json"
SUMMARY_PATH = config.OUTPUT_DIR / "summary.md"


def _cost(usage: dict) -> float:
    return (usage.get("prompt_tokens", 0) / 1e6 * config.COST_PER_1M_INPUT
            + usage.get("completion_tokens", 0) / 1e6 * config.COST_PER_1M_OUTPUT)


def _tfidf_similarity(a: str, b: str) -> float:
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


def _embedding_similarities(rows: list[dict], desc_lookup: dict[str, str]) -> list[float] | None:
    """Batch-encode every description and official text once; return a cosine
    similarity list aligned with `rows` (each row is one generation)."""
    try:
        from src import embed
        gen_emb = embed.encode([r["description"] for r in rows])
        off_emb = embed.encode([desc_lookup.get(r["id"], "") for r in rows])
        sims = embed.cosine_from_embeddings(gen_emb, off_emb)
        return [round(float(s), 4) for s in sims]
    except Exception as e:  # noqa: BLE001
        print(f"  !! embedding similarity unavailable: {e}")
        return None


def evaluate(rows: list[dict] | None = None, use_embeddings: bool = True) -> list[dict]:
    rows = rows if rows is not None else load_generated()

    desc_lookup: dict[str, str] = {}
    for p in config.PROFILE_DIR.glob("*.json"):
        try:
            prof = json.loads(p.read_text(encoding="utf-8"))
            desc_lookup[prof["id"]] = prof.get("description") or ""
        except Exception:  # noqa: BLE001
            continue

    emb_sims = _embedding_similarities(rows, desc_lookup) if use_embeddings else None

    results: list[dict] = []
    for idx, r in enumerate(rows):
        usage = r.get("usage", {})
        official = desc_lookup.get(r["id"], "")
        emb = emb_sims[idx] if (official and emb_sims is not None) else None
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
            "tfidf_similarity": round(_tfidf_similarity(r["description"], official), 4)
                if official else None,
            "embedding_similarity": emb,
            "has_official_desc": bool(official),
        })

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(results: list[dict]) -> str:
    from collections import defaultdict
    import numpy as np

    by_strat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_strat[r["strategy"]].append(r)

    lines = ["# Evaluation Summary", "",
             f"Total generations: {len(results)}", ""]
    for strat in ["baseline", "content", "semantic"]:
        rs = by_strat.get(strat, [])
        if not rs:
            continue
        tf = [r["tfidf_similarity"] for r in rs if r["tfidf_similarity"] is not None]
        em = [r["embedding_similarity"] for r in rs if r["embedding_similarity"] is not None]
        lines += [
            f"## {strat}  (n={len(rs)})",
            f"- avg total tokens: {_avg([r['total_tokens'] for r in rs]):.1f}",
            f"- avg elapsed: {_avg([r['elapsed_seconds'] for r in rs]):.2f}s",
            f"- avg est. cost: ${_avg([r['estimated_cost_usd'] for r in rs]):.6f}",
            f"- avg TF-IDF similarity: {_avg(tf):.4f}  (n={len(tf)})",
            f"- avg embedding similarity: {_avg(em):.4f}  (n={len(em)})",
            "",
        ]

    # Agreement between the two metrics (alignment check).
    pairs = [(r["tfidf_similarity"], r["embedding_similarity"])
             for r in results
             if r["tfidf_similarity"] is not None and r["embedding_similarity"] is not None]
    if len(pairs) >= 2:
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        corr = float(np.corrcoef(a, b)[0, 1])
        lines += ["## Metric agreement",
                  f"- Pearson correlation (TF-IDF vs embedding): {corr:.4f}",
                  f"- n={len(pairs)} paired observations", ""]

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)
