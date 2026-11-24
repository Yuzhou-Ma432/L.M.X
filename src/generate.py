"""Generate dataset descriptions under the three input strategies.

    baseline    — title + column names only
    content     — title + columns + dtypes + samples + summary statistics
    semantic    — content, plus a first LLM pass that summarizes likely topic/use

Each strategy writes `data/outputs/generated/{id}/{strategy}.json` so results
can be inspected and the evaluation stage can aggregate token/time/cost.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from src import llm

SYSTEM_PROMPT = (
    "You are an expert data curator writing catalog descriptions for tabular "
    "datasets. Write a concise, accurate, plain-language description of what "
    "the dataset contains and how it could be used. Do not mention that you are "
    "an AI. Return only the description text, with no heading, preamble, or "
    "markdown formatting."
)


def _col_names(profile: dict) -> str:
    return ", ".join(c["name"] for c in profile["columns"])


def _col_details(profile: dict) -> str:
    lines = []
    for c in profile["columns"]:
        bits = [f"- {c['name']} (type {c['dtype']})",
                f"{c['nunique']} distinct values",
                f"{c['missing_pct']}% missing"]
        if c.get("stats"):
            s = c["stats"]
            bits.append(f"min={s.get('min')} max={s.get('max')} mean={s.get('mean')} "
                        f"p25={s.get('p25')} p75={s.get('p75')}")
        if c.get("top_values"):
            tv = ", ".join(f"{t['value']} ({t['pct']}%)" for t in c["top_values"])
            bits.append("common values: " + tv)
        if c.get("sample_values"):
            bits.append("samples: " + ", ".join(c["sample_values"]))
        lines.append(" ".join(bits))
    return "\n".join(lines)


def _describe_title(profile: dict) -> str:
    return (
        f"Title: {profile['name']}\n"
        f"Category: {profile.get('category', 'unknown')}\n"
        f"Rows (sampled): {profile.get('n_rows', '?')}\n"
        f"Columns: {profile.get('n_cols', '?')}\n"
    )


def build_baseline_prompt(profile: dict) -> str:
    return (
        f"{_describe_title(profile)}"
        f"Column names: {_col_names(profile)}\n\n"
        "Write a clear description of this dataset."
    )


def build_content_prompt(profile: dict) -> str:
    return (
        f"{_describe_title(profile)}\n"
        f"Column details:\n{_col_details(profile)}\n\n"
        "Write a clear description of this dataset."
    )


def build_semantic_prompt(profile: dict) -> str:
    return (
        "Given the following metadata about a tabular dataset, summarize in a "
        "few sentences its likely topic, the domain it belongs to, and the "
        "possible uses of this data.\n\n"
        f"{_describe_title(profile)}\n"
        f"Column names: {_col_names(profile)}\n"
    )


def build_semantic_enhanced_prompt(profile: dict, semantic_summary: str) -> str:
    return (
        f"{_describe_title(profile)}\n"
        f"Column details:\n{_col_details(profile)}\n\n"
        f"Semantic summary (topic and possible uses):\n{semantic_summary}\n\n"
        "Write a clear description of this dataset."
    )


def _save(profile: dict, strategy: str, result: dict) -> None:
    d = config.GENERATED_DIR / profile["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{strategy}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def generate_one(profile: dict, strategy: str) -> dict | None:
    t0 = time.time()

    if strategy == "baseline":
        prompt = build_baseline_prompt(profile)
        extra = None
    elif strategy == "content":
        prompt = build_content_prompt(profile)
        extra = None
    elif strategy == "semantic":
        semantic = llm.chat_with_retry(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": build_semantic_prompt(profile)}],
            max_tokens=config.SEMANTIC_MAX_TOKENS,
        )
        prompt = build_semantic_enhanced_prompt(profile, semantic["content"])
        extra = {"semantic_summary": semantic["content"],
                 "semantic_usage": {k: semantic[k] for k in
                                    ("prompt_tokens", "completion_tokens", "total_tokens")}}
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    final = llm.chat_with_retry(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        max_tokens=config.MAX_DESCRIPTION_TOKENS,
    )

    result = {
        "id": profile["id"],
        "name": profile["name"],
        "strategy": strategy,
        "description": final["content"],
        "prompt": prompt,
        "model": final["model"],
        "finish_reason": final["finish_reason"],
        "usage": {k: final[k] for k in
                  ("prompt_tokens", "completion_tokens", "total_tokens")},
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    if extra:
        result.update(extra)
    _save(profile, strategy, result)
    return result


STRATEGIES = ["baseline", "content", "semantic"]


def _gen_task(profile: dict, strat: str) -> dict | None:
    try:
        r = generate_one(profile, strat)
        print(f"  {profile['id']} :: {strat} -> {r['usage']['total_tokens']} tok / {r['elapsed_seconds']}s")
        return r
    except Exception as e:  # noqa: BLE001 — a failed dataset must not stop the run
        print(f"  {profile['id']} :: {strat} !! failed: {e}")
        return None


def generate_all(profiles: list[dict], strategies: list[str] | None = None,
                 workers: int = 1) -> list[dict]:
    strategies = strategies or STRATEGIES
    tasks = [(p, s) for p in profiles for s in strategies]
    total = len(tasks)
    results: list[dict] = []

    if workers <= 1:
        for i, (profile, strat) in enumerate(tasks, 1):
            print(f"[{i}/{total}] {profile['id']} :: {strat}")
            r = _gen_task(profile, strat)
            if r:
                results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_gen_task, p, s): (p, s) for p, s in tasks}
            done = 0
            for fut in as_completed(futs):
                done += 1
                p, s = futs[fut]
                r = fut.result()
                if r:
                    results.append(r)
                if done % 20 == 0:
                    print(f"  progress: {done}/{total}")

    print(f"Generated {len(results)}/{total} descriptions.")
    return results
