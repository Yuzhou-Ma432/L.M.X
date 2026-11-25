"""AutoDDG generation steps, using AutoDDG's own prompts verbatim.

semantic_profile -> UFD (user-focused description) -> topic -> SFD (search-
focused description). Only the prompt *inputs* are changed by the
experiments; the templates are untouched.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

PROMPTS_PATH = Path(__file__).parent / "prompts" / "autoddg_prompts.yaml"
DESCRIPTION_WORDS = 100


@lru_cache(maxsize=1)
def prompts() -> dict:
    return yaml.safe_load(PROMPTS_PATH.read_text())


# ---------------------------------------------------------------- semantic --

def semantic_profile(llm, column_data: list[tuple[str, list[str]]]) -> dict[str, dict]:
    """Group-prompt semantic profiler (AutoDDG `_get_semantic_types_group`)."""
    p = prompts()["semantic_profiler"]
    columns_text = "\n\n".join(f"Column: {n}\nSample values: {v}" for n, v in column_data)
    user = p["group_user_prompt"].format(template=p["template"],
                                         response_example=p["group_response_example"],
                                         columns_text=columns_text)
    system = ("You are a helpful assistant skilled in dataset semantic analysis. You analyze "
              "multiple columns efficiently in a single response.")
    raw = llm.chat(system, user, max_tokens=4000)
    return _parse_json(raw)


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            out = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0))) if m else {}
        except json.JSONDecodeError:
            out = {}
    return out if isinstance(out, dict) else {}


def _yes(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() in {"yes", "true"})


def render_semantic(sem: dict[str, dict], hide_columns: set[str] | None = None) -> str:
    """AutoDDG `_create_column_summary`, joined."""
    hide_columns = hide_columns or set()
    lines = []
    for col, d in sem.items():
        if col in hide_columns or not isinstance(d, dict):
            continue
        s = f"**{col}**: "
        et = d.get("Entity Type")
        if isinstance(et, str) and et.lower() not in {"", "unknown"}:
            s += f"Represents {et.lower()}. "
        t = d.get("Temporal", {})
        if isinstance(t, dict) and _yes(t.get("isTemporal")):
            s += f"Contains temporal data (resolution: {t.get('resolution', 'unknown')}). "
        sp = d.get("Spatial", {})
        if isinstance(sp, dict) and _yes(sp.get("isSpatial")):
            s += f"Contains spatial data (resolution: {sp.get('resolution', 'unknown')}). "
        dt = d.get("Domain-Specific Types")
        if isinstance(dt, str) and dt.lower() not in {"", "unknown"}:
            s += f"Domain-specific type: {dt.lower()}. "
        fc = d.get("Function/Usage Context")
        if isinstance(fc, str) and fc.lower() not in {"", "unknown"}:
            s += f"Function/Usage context: {fc.lower()}. "
        lines.append(s.strip())
    return "\n".join(lines)


# ------------------------------------------------------------- descriptions --

def dataset_sample_text(sample: pd.DataFrame, title: str | None = None) -> str:
    """CSV sample shown to the UFD/topic prompts. Unlike stock AutoDDG we prefix
    the portal title (portals always have one; the attacker controls it too)."""
    csv = sample.to_csv(index=False).strip()
    return (f"Title: {title}\n" if title else "") + csv


def ufd(llm, sample_text: str, profile_text: str | None, semantic_text: str | None,
        topic: str | None = None) -> str:
    p = prompts()["dataset_description"]
    parts = [p["introduction"].format(dataset_sample=sample_text)]
    if profile_text:
        parts.append(p["profile_instruction"].format(dataset_profile=profile_text))
    if semantic_text:
        parts.append(p["semantic_instruction"].format(semantic_profile=semantic_text))
    if topic:
        parts.append(p["topic_instruction"].format(data_topic=topic))
    parts += [p["closing_instruction"], f"Target length: approximately {DESCRIPTION_WORDS} words."]
    return llm.chat(p["system_message"].strip(), "\n".join(parts), max_tokens=400)


def topic(llm, title: str, sample_text: str) -> str:
    p = prompts()["topic_generation"]
    user = p["user_prompt"].format(title=title, description_block="", dataset_sample=sample_text)
    return llm.chat(p["system_message"].strip(), user, max_tokens=20).strip().strip('."')


def sfd(llm, initial_description: str, topic_: str) -> str:
    p = prompts()["search_expansion"]
    user = p["user_prompt"].format(topic=topic_, initial_description=initial_description,
                                   template=p["template"])
    return llm.chat(p["system_message"].strip(), user, max_tokens=900)
