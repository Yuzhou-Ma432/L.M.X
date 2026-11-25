"""LLM clients: an OpenAI-compatible client with an on-disk cache, and an
offline MockLLM used to exercise the pipeline without an API key.

Every client exposes ``chat(system, user, max_tokens) -> str``.

The MockLLM is NOT a model of any real LLM. It is a deterministic simulator of
a worst case: it copies every fact it is shown into the description, and it
obeys instructions embedded in the data. Numbers produced with it validate the
plumbing and the defenses' mechanics only; real findings need a real model.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

CACHE_DIR = Path(os.getenv("ROBUSTDDG_CACHE", ".cache/llm"))


class OpenAICompatClient:
    """Works with DeepSeek, OpenAI, vLLM, Ollama, ... (any /chat/completions)."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 cache: bool = True):
        from openai import OpenAI  # lazy: only needed for real runs

        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
            api_key=api_key or os.getenv("LLM_API_KEY"),
        )
        self.temperature = temperature
        self.cache = cache
        self.calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _key(self, system: str, user: str, max_tokens: int) -> Path:
        h = hashlib.sha256(json.dumps(
            [self.model, self.temperature, max_tokens, system, user]).encode()).hexdigest()
        return CACHE_DIR / f"{h}.json"

    def chat(self, system: str, user: str, max_tokens: int = 400) -> str:
        path = self._key(system, user, max_tokens)
        if self.cache and path.exists():
            return json.loads(path.read_text())["content"]
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                break
            except Exception:  # noqa: BLE001 — retry transient API errors
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        content = (resp.choices[0].message.content or "").strip()
        self.calls += 1
        if resp.usage:
            self.usage["prompt_tokens"] += resp.usage.prompt_tokens
            self.usage["completion_tokens"] += resp.usage.completion_tokens
        if self.cache:
            path.write_text(json.dumps({"content": content}))
        return content


# --------------------------------------------------------------------------
# Offline simulator
# --------------------------------------------------------------------------

_INJECTION = re.compile(
    r"(?:ignore|disregard|forget)\b.{0,40}?\b(?:previous|above|prior|earlier)\b.{0,80}?"
    r"(?:describe|summari[sz]e|present|call|label)\s+(?:this|the)\s+(?:dataset|data|table)?\s*"
    r"as\s+(?:an?\s+)?(?P<target>[^.;\"'\n]{3,80})",
    re.I | re.S)

_STOP = set("""a an the of and or for to in on at by with from is are was were be this that
these those it its as data dataset table column columns values value rows row sample
there unique missing type mean standard deviation minimum maximum key profile
information includes represents contains resolution domain specific function usage
context entity semantic types temporal spatial""".split())


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower()) if w not in _STOP]


class MockLLM:
    """Deterministic stand-in: an obedient, faithful copier (see module doc).

    ``compliance`` is the fraction of injected instructions it obeys, decided
    by a hash of the prompt so runs are reproducible.
    """

    model = "mock"

    def __init__(self, compliance: float = 1.0):
        self.compliance = compliance
        self.calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def _obeys(self, text: str) -> bool:
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return h < self.compliance

    def _hijack(self, user: str) -> str | None:
        m = _INJECTION.search(user)
        if m and self._obeys(user):
            return m.group("target").strip()
        return None

    def chat(self, system: str, user: str, max_tokens: int = 400) -> str:
        self.calls += 1
        if "semantic analyzer" in user:
            return self._semantic(user)
        if "concise topic" in user or "Topic (2-3 words)" in user:
            return self._topic(user)
        if "expand the description" in user:
            return self._sfd(user)
        return self._ufd(user)

    # -- semantic profiler ---------------------------------------------------
    def _semantic(self, user: str) -> str:
        cols = re.findall(r"Column(?: name)?: (.*?)\nSample values: (.*?)(?:\n|$)", user)
        out = {}
        for name, samples in cols:
            n = name.lower()
            temporal = re.search(r"year|date|time|month|day|period|fy\b", n)
            if not temporal and re.fullmatch(r"\[?['\"]?(19|20)\d\d", samples[:6].strip("[' \"")):
                temporal = True
            spatial = re.search(r"(^|_)(lat|latitude|lon|lng|longitude)($|_)|borough|boro|zip|address|city|state|county|"
                                r"community|district|precinct|bbl|bin\b|x_coord|y_coord", n)
            res_t = ("Year" if re.search(r"year|fy\b", n) else "Day") if temporal else ""
            res_s = ("Coordinates" if re.search(r"(^|_)(lat|latitude|lon|lng|longitude)($|_)|coord", n) else
                     "Zip Code" if "zip" in n else "City District") if spatial else ""
            if re.search(r"(^|_)id$|number$|code$|bbl|bin\b", n):
                ent, fn = "Identifier", "Aggregation Key"
            elif re.search(r"count|total|amount|rate|pct|percent|number_of|num_|avg|sum", n):
                ent, fn = "Numeric Entity", "Measurement"
            elif temporal:
                ent, fn = "Temporal Entity", "Aggregation Key"
            elif spatial:
                ent, fn = "Location", "Aggregation Key"
            else:
                ent, fn = "Category", "Categorical Attribute"
            out[name] = {"Temporal": {"isTemporal": bool(temporal), "resolution": res_t},
                         "Spatial": {"isSpatial": bool(spatial), "resolution": res_s},
                         "Entity Type": ent, "Domain-Specific Types": "General",
                         "Function/Usage Context": fn}
        return json.dumps(out)

    # -- unified description ---------------------------------------------------
    def _ufd(self, user: str) -> str:
        target = self._hijack(user)
        if target:
            return (f"This dataset is a {target} dataset. It provides comprehensive records "
                    f"about {target}, useful for researchers and policymakers studying {target}.")
        title = re.search(r"Title: (.*)", user)
        title = title.group(1).strip() if title else "this table"
        sents = [f"The {title} dataset is a tabular dataset."]
        hdr = re.search(r"Rows: (\d+).*?Columns: (\d+)", user)
        if hdr:
            sents.append(f"It contains {hdr.group(1)} rows and {hdr.group(2)} columns.")
        # Faithful copier: every per-column profile line becomes a sentence.
        for name, body in re.findall(r"\*\*(.+?)\*\*: (.+)", user):
            if "Represents" in body and "Data is of type" not in body:
                continue  # semantic line; summarized below
            facts = re.sub(r"\s+", " ", body).strip()
            sents.append(f"The column {name} has these properties: {facts}")
        sem = re.findall(r"\*\*(.+?)\*\*: (Represents [^\n]+)", user)
        for name, body in sem[:8]:
            sents.append(f"Semantically, {name} {body[0].lower() + body[1:]}")
        # Copy the first sample row (the "dataset sample" section) as-is.
        sample = re.search(r"dataset sample:\s*\n+(.+?)\n\n", user, re.S)
        if sample:
            first = sample.group(1).splitlines()[:3]
            sents.append("Example records include: " + " | ".join(first))
        return " ".join(sents)

    def _topic(self, user: str) -> str:
        target = self._hijack(user)
        if target:
            return " ".join(target.split()[:3]).title()
        title = re.search(r"Title: (.*)", user)
        return " ".join((title.group(1) if title else "General Data").split()[:3]).title()

    def _sfd(self, user: str) -> str:
        target = self._hijack(user)
        topic = re.search(r"about the topic (.*?), with", user)
        topic = topic.group(1) if topic else ""
        init = re.search(r"initial description:\s*\n+(.*?)\n\nPlease expand", user, re.S)
        init = init.group(1).strip() if init else ""
        words = list(dict.fromkeys(_words(init)))
        if target:
            words = _words(target) * 3 + words
        kw = "\n".join(f"- {w}" for w in words[:40])
        return (f"Dataset Overview:\n{init}\n\nKey Themes or Topics:\n- {topic}\n"
                f"{('- ' + target) if target else ''}\n\nKeywords and Themes:\n{kw}")
