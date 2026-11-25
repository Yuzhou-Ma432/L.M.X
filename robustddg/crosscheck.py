"""Cross-check stage: the statistical profile and the LLM's column semantics
must agree, otherwise the disputed fact is withheld from the generation prompt
(flagged, never guessed).

Two independent evidence sources are compared against the derived statistics:

* raw sample values (what the profiler read, before any derivation), and
* the semantic profile (the LLM's reading of column name + samples).

Each rule returns ``Flag(column, field, rule, message)`` where ``field`` is the
fact group to withhold: ``range`` | ``unit`` | ``type``.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd

from .profiler import infer_unit

YEAR_MIN, YEAR_MAX = 1800, 2100   # for 'year'-typed numbers
DATE_MIN = 1600                    # for datetimes (NYC has 1686 deeds)
_NUM = re.compile(r"^-?\$?[\d,]*\.?\d+(?:[eE][-+]?\d+)?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}|^\d{1,2}/\d{1,2}/\d{2,4}")


@dataclass
class Flag:
    column: str
    field: str
    rule: str
    message: str

    def dict(self) -> dict:
        return asdict(self)


def _nums(samples: list[str]) -> list[float]:
    out = []
    for s in samples:
        s = s.strip()
        if s and _NUM.match(s):
            out.append(float(s.replace("$", "").replace(",", "")))
    return out


def _dates(samples: list[str]) -> list[pd.Timestamp]:
    out = []
    for s in samples:
        if s and _DATE.match(s.strip()):
            t = pd.to_datetime(s, errors="coerce")
            if pd.notna(t):
                out.append(t.tz_localize(None) if t.tzinfo else t)
    return out


def _yes(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() in {"yes", "true"})


def _money_cue(name: str, samples: list[str], sem: dict) -> bool:
    dom = str(sem.get("Domain-Specific Types", "")).lower()
    return (infer_unit(name) == "US dollars" or "financ" in dom
            or any(s.strip().startswith("$") for s in samples))


def check_column(c: dict, sem: dict | None) -> list[Flag]:
    sem = sem or {}
    name, typ, samples = c["name"], c["type"], [s for s in c.get("samples", []) if s]
    flags: list[Flag] = []
    nums, dates = _nums(samples), _dates(samples)
    temporal = sem.get("Temporal", {}) if isinstance(sem.get("Temporal"), dict) else {}
    spatial = sem.get("Spatial", {}) if isinstance(sem.get("Spatial"), dict) else {}
    is_temporal = _yes(temporal.get("isTemporal"))
    res_t = str(temporal.get("resolution", "")).lower()

    # --- 1. type: derived type vs what the raw samples look like ----------------
    if samples:
        if typ in ("integer", "float") and len(nums) < 0.5 * len(samples):
            flags.append(Flag(name, "type", "samples_not_numeric",
                              f"profiled as {typ} but samples look like {samples[:2]}"))
        if typ == "datetime" and len(dates) < 0.5 * len(samples):
            flags.append(Flag(name, "type", "samples_not_dates",
                              f"profiled as datetime but samples look like {samples[:2]}"))
        if typ in ("integer", "float", "categorical", "text") and len(dates) >= 0.8 * len(samples):
            flags.append(Flag(name, "type", "samples_are_dates",
                              f"profiled as {typ} but samples are dates"))

    # --- 2. range: raw samples must lie inside the derived range ---------------
    if typ in ("integer", "float") and "min" in c and nums:
        lo, hi = c["min"], c["max"]
        tol = 1e-3 * max(abs(lo), abs(hi), 1)
        outside = [x for x in nums if x < lo - tol or x > hi + tol]
        if outside:
            flags.append(Flag(name, "range", "sample_outside_range",
                              f"sample {outside[0]:g} outside reported [{lo:g}, {hi:g}]"))
        else:
            # Magnitude check (robust to skew: median vs median) — catches x1000
            # scale errors when min is 0 and every sample is still "in range".
            absn = sorted(abs(x) for x in nums)
            med = absn[len(absn) // 2] or sum(absn) / len(absn)  # mean if median is 0
            ref = abs(c.get("median", c.get("mean", 0))) or abs(c.get("mean", 0))
            if med > 0 and ref > 0 and not (0.01 <= ref / med <= 100):
                flags.append(Flag(name, "range", "magnitude_mismatch",
                                  f"reported median {ref:g} vs sample median {med:g}"))
    if typ == "datetime" and "date_min" in c and dates:
        lo, hi = pd.Timestamp(c["date_min"]), pd.Timestamp(c["date_max"]) + pd.Timedelta(days=1)
        outside = [d for d in dates if d < lo or d > hi]
        if outside:
            flags.append(Flag(name, "range", "sample_outside_date_range",
                              f"sample {outside[0].date()} outside reported "
                              f"{c['date_min']}..{c['date_max']}"))

    # --- 3. semantics (LLM) vs statistics --------------------------------------
    if is_temporal and "year" in res_t and typ in ("integer", "float") and "min" in c:
        if c["min"] < YEAR_MIN or c["max"] > YEAR_MAX:
            flags.append(Flag(name, "range", "year_out_of_bounds",
                              f"LLM says year, but values span {c['min']:g}..{c['max']:g}"))
    if typ == "datetime" and "date_min" in c:
        y0, y1 = int(c["date_min"][:4]), int(c["date_max"][:4])
        if y0 < DATE_MIN or y1 > YEAR_MAX:
            flags.append(Flag(name, "range", "date_out_of_bounds",
                              f"date range {c['date_min']}..{c['date_max']} implausible"))
        if sem and not is_temporal:
            flags.append(Flag(name, "type", "llm_not_temporal",
                              "profiled as datetime but LLM says not temporal"))
    if _yes(spatial.get("isSpatial")) and "min" in c:
        n = name.lower()
        bound = (90 if re.search(r"(^|_)lat(itude)?($|_)", n) else
                 180 if re.search(r"(^|_)(lon|lng|long|longitude)($|_)", n) else None)
        if bound and (c["min"] < -bound or c["max"] > bound):
            flags.append(Flag(name, "range", "coordinate_out_of_bounds",
                              f"LLM says coordinates but values span {c['min']:g}..{c['max']:g}"))

    # --- 4. unit vs semantics/statistics --------------------------------------
    unit = c.get("unit")
    if unit and "min" in c:
        if unit == "percent" and (c["max"] > 100 or c["min"] < 0):
            flags.append(Flag(name, "unit", "percent_out_of_bounds",
                              f"unit percent but values span {c['min']:g}..{c['max']:g}"))
        if unit == "US dollars" and not _money_cue(name, samples, sem):
            flags.append(Flag(name, "unit", "money_without_cue",
                              "unit US dollars but neither name, samples nor LLM domain is monetary"))
        if unit == "count" and (c["min"] < 0 or c["type"] == "float"):
            flags.append(Flag(name, "unit", "count_not_nonneg_int",
                              "unit count but values are negative or fractional"))
        if unit.startswith("degrees l") and (abs(c["min"]) > 180 or abs(c["max"]) > 180):
            flags.append(Flag(name, "unit", "degrees_out_of_bounds", "coordinate unit out of bounds"))
        expected = infer_unit(name)
        if expected and expected != unit and unit != "count":
            flags.append(Flag(name, "unit", "unit_name_mismatch",
                              f"unit {unit} but column name suggests {expected}"))
    return flags


def crosscheck(profile: dict, semantic: dict[str, dict]) -> list[Flag]:
    flags: list[Flag] = []
    for c in profile["columns"]:
        flags += check_column(c, semantic.get(c["name"]))
    return flags


def withheld_map(flags: list[Flag]) -> dict[str, dict[str, str]]:
    """Flags -> {column: {field: reason}} for `profiler.render_profile`."""
    out: dict[str, dict[str, str]] = {}
    for f in flags:
        out.setdefault(f.column, {}).setdefault(
            f.field, "withheld: profiler and semantic evidence disagree")
    return out
