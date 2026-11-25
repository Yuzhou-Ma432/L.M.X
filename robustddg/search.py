"""BM25 dataset search over (search-focused) descriptions + ranking metrics.

Self-contained Okapi BM25 so the prototype has no nltk/rank_bm25 dependency;
it mirrors AutoDDG's `SimpleSearchEngine(similarity="BM25")`.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_STOP = set("""a about above after all also an and any are as at be been being but by can
could did do does for from had has have how i if in into is it its may more most no not of
on or other our out over same should so some such than that the their them then there these
they this those through to too under up very was we were what when where which while who
will with within would you your""".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    out = []
    for t in toks:
        if t in _STOP or len(t) < 2:
            continue
        out.append(t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t)
        if "-" in t:  # "covid-19" also indexes "covid", "19"
            out.extend(p for p in t.split("-") if p and p not in _STOP)
    return out


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.ids = list(docs)
        self.toks = [tokenize(docs[i]) for i in self.ids]
        self.tf = [Counter(t) for t in self.toks]
        self.len = [len(t) for t in self.toks]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(w for t in self.tf for w in t)
        n = len(self.ids)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> dict[str, float]:
        q = tokenize(query)
        out = {}
        for i, (tf, L) in enumerate(zip(self.tf, self.len)):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out[self.ids[i]] = s
        return out

    def ranking(self, query: str) -> list[str]:
        sc = self.scores(query)
        # Ties broken by id for determinism; zero-score docs are "not retrieved".
        return [d for d, s in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0])) if s > 0]

    def rank_of(self, doc_id: str, query: str) -> int | None:
        r = self.ranking(query)
        return r.index(doc_id) + 1 if doc_id in r else None


def rr(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def hit(rank: int | None, k: int) -> float:
    return 1.0 if rank and rank <= k else 0.0


def ndcg_at_k(ranking: list[str], rel: dict[str, float], k: int = 10) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
