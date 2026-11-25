"""Controlled profiler-error injection with detectable probes.

Each operator corrupts one *derived* statistic of one column (raw samples are
left intact, as with a real profiler bug) and returns an ``Injection`` record
holding a ``fake`` signal (what would appear in text if the error propagated)
and a ``true`` signal (what a correct description might say).

Operators — modelled on real profiler failure modes:

    date_range  wrong temporal coverage (dates/years shifted by -37 years)
    scale       wrong magnitude (x1000, e.g. "in thousands" mishandled)
    unit        wrong unit label (count -> US dollars, dollars -> percent, ...)
    type        wrong type (integer misparsed as Unix-epoch datetime, or
                datetime collapsed to epoch-second integers)
"""
from __future__ import annotations

import copy
import hashlib
import random
import re
from dataclasses import asdict, dataclass, field

import pandas as pd

OPS = ("date_range", "scale", "unit", "type")
YEAR_SHIFT = 37


@dataclass
class Injection:
    op: str
    column: str
    field: str                         # range | unit | type
    fake: dict = field(default_factory=dict)   # {"years"|"numbers"|"terms": [...]}
    true: dict = field(default_factory=dict)
    detail: str = ""

    def dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------- probes --

_MULT = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mn": 1e6,
         "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12}
_NUMTOK = re.compile(r"(-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)\s*(thousand|million|billion|trillion|"
                     r"mn|bn|k|m|b)?\b", re.I)
UNIT_TERMS = {
    "US dollars": r"\$|\bdollars?\b|\busd\b",
    "percent": r"%|\bpercent(?:age)?s?\b",
    "square feet": r"square\s+f(?:ee|oo)t|\bsq\.?\s*ft\b",
    "acres": r"\bacres?\b",
    "miles": r"\bmiles?\b",
    "degrees Fahrenheit": r"fahrenheit|°f\b",
    "kilograms": r"kilograms?|\bkg\b",
}


def years_in(text: str) -> set[int]:
    return {int(y) for y in re.findall(r"(?<!\d)(1[89]\d\d|20\d\d|2100)(?!\d)", text)}


def numbers_in(text: str) -> list[float]:
    out = []
    for num, mult in _NUMTOK.findall(text):
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        out.append(v * _MULT.get(mult.lower(), 1) if mult else v)
    return out


def present(text: str, sig: dict) -> bool:
    if not sig:
        return False
    if sig.get("years") and years_in(text) & set(sig["years"]):
        return True
    if sig.get("numbers"):
        nums = numbers_in(text)
        for t in sig["numbers"]:
            if any(abs(x - t) <= 0.005 * abs(t) for x in nums):
                return True
    for pat in sig.get("terms", []):
        if re.search(pat, text, re.I):
            return True
    return False


# --------------------------------------------------------------- operators --

def _is_year_col(c: dict) -> bool:
    return (c["type"] == "integer" and "min" in c and 1800 <= c["min"] and c["max"] <= 2100
            and re.search(r"year|yr|fy", c["name"], re.I) is not None)


def _nontrivial_numeric(c: dict) -> bool:
    return (c["type"] in ("integer", "float") and "min" in c and not _is_year_col(c)
            and c["n_distinct"] > 2 and max(abs(c["min"]), abs(c["max"])) > 0)


def _sig_numbers(*vals: float) -> list[float]:
    return [v for v in vals if v not in (0, 1, -1) and abs(v) >= 2]


def op_date_range(p: dict, rng: random.Random) -> Injection | None:
    cands = [c for c in p["columns"] if (c["type"] == "datetime" and "date_min" in c)
             or _is_year_col(c)]
    if not cands:
        return None
    c = rng.choice(cands)
    if c["type"] == "datetime":
        y0, y1 = int(c["date_min"][:4]), int(c["date_max"][:4])
        sh = lambda d: (pd.Timestamp(d) - pd.DateOffset(years=YEAR_SHIFT)).strftime("%Y-%m-%d")
        c["date_min"], c["date_max"] = sh(c["date_min"]), sh(c["date_max"])
    else:
        y0, y1 = int(c["min"]), int(c["max"])
        for k in ("min", "max", "mean", "median"):
            c[k] -= YEAR_SHIFT
    return Injection("date_range", c["name"], "range",
                     fake={"years": sorted({y0 - YEAR_SHIFT, y1 - YEAR_SHIFT})},
                     true={"years": sorted({y0, y1})},
                     detail=f"{y0}-{y1} -> {y0 - YEAR_SHIFT}-{y1 - YEAR_SHIFT}")


def op_scale(p: dict, rng: random.Random) -> Injection | None:
    cands = [c for c in p["columns"] if _nontrivial_numeric(c)]
    if not cands:
        return None
    c = rng.choice(cands)
    true = _sig_numbers(c["min"], c["max"], c["mean"])
    for k in ("min", "max", "mean", "median", "std"):
        c[k] = float(f"{c[k] * 1000:.4g}")
    fake = _sig_numbers(c["min"], c["max"], c["mean"])
    if not fake:
        return None
    return Injection("scale", c["name"], "range", fake={"numbers": fake},
                     true={"numbers": true}, detail="x1000")


_WRONG_UNIT = {None: "US dollars", "count": "US dollars", "US dollars": "percent",
               "percent": "US dollars"}


def op_unit(p: dict, rng: random.Random) -> Injection | None:
    cands = [c for c in p["columns"] if _nontrivial_numeric(c)]
    if not cands:
        return None
    c = rng.choice(cands)
    old = c.get("unit")
    new = _WRONG_UNIT.get(old, "kilograms")
    c["unit"] = new
    return Injection("unit", c["name"], "unit", fake={"terms": [UNIT_TERMS[new]]},
                     true={"terms": [UNIT_TERMS[old]]} if old in UNIT_TERMS else {},
                     detail=f"{old} -> {new}")


def op_type(p: dict, rng: random.Random) -> Injection | None:
    ints = [c for c in p["columns"] if c["type"] == "integer" and _nontrivial_numeric(c)
            and c["max"] < 1e7]
    dts = [c for c in p["columns"] if c["type"] == "datetime" and "date_min" in c]
    if ints:
        c = rng.choice(ints)
        true = _sig_numbers(c["min"], c["max"], c["mean"])
        lo = pd.to_datetime(c["min"], unit="s").strftime("%Y-%m-%d")
        hi = pd.to_datetime(c["max"], unit="s").strftime("%Y-%m-%d")
        for k in ("min", "max", "mean", "median", "std"):
            c.pop(k, None)
        c.update(type="datetime", date_min=lo, date_max=hi)
        c["unit"] = None
        return Injection("type", c["name"], "type", fake={"years": [1970]},
                         true={"numbers": true}, detail=f"integer -> epoch datetime {lo}..{hi}")
    if dts:
        c = rng.choice(dts)
        y0, y1 = int(c["date_min"][:4]), int(c["date_max"][:4])
        lo = pd.Timestamp(c.pop("date_min")).timestamp()
        hi = pd.Timestamp(c.pop("date_max")).timestamp()
        mid = float(f"{(lo + hi) / 2:.4g}")
        c.update(type="integer", min=float(f"{lo:.4g}"), max=float(f"{hi:.4g}"),
                 mean=mid, median=mid, std=float(f"{(hi - lo) / 4:.4g}"))
        return Injection("type", c["name"], "type",
                         fake={"numbers": _sig_numbers(c["min"], c["max"], c["mean"])},
                         true={"years": sorted({y0, y1})}, detail="datetime -> epoch seconds")
    return None


_OPS = {"date_range": op_date_range, "scale": op_scale, "unit": op_unit, "type": op_type}


def inject(profile: dict, op: str, seed_key: str = "") -> tuple[dict, Injection] | None:
    """Return (corrupted_copy, record), or None if the op is not applicable."""
    p = copy.deepcopy(profile)
    seed = int(hashlib.md5(f"{seed_key}:{op}".encode()).hexdigest()[:8], 16)
    rec = _OPS[op](p, random.Random(seed))
    return (p, rec) if rec else None
