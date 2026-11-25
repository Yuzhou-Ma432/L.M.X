"""End-to-end AutoDDG-style pipeline with hooks for both prototypes.

    run(df, title, llm,
        crosscheck=False,          # prototype 1: withhold disputed profile facts
        defenses=(),               # prototype 2: {"sanitize","stats_only","consistency"}
        profile=None,              # override (e.g. a corrupted profile)
        semantic=None)             # reuse a semantic profile across conditions

Order (as in AutoDDG): statistical profile, semantic profile, topic, UFD, SFD.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from . import defenses as D
from . import generate as G
from .crosscheck import crosscheck as run_crosscheck
from .crosscheck import withheld_map
from .profiler import profile_df, render_profile, sample_rows


@dataclass
class Result:
    title: str
    topic: str
    ufd: str
    sfd: str
    profile_text: str
    semantic_text: str
    semantic: dict
    flags: list[dict] = field(default_factory=list)
    defense_report: dict = field(default_factory=dict)

    def dict(self) -> dict:
        return asdict(self)


def semantic_for(llm, df: pd.DataFrame, hide_samples: set[str] = frozenset()) -> dict:
    samp = sample_rows(df)
    cols = [(str(c), [] if str(c) in hide_samples else samp[c].astype(object).where(samp[c].notna(), "").astype(str).tolist())
            for c in df.columns]
    return G.semantic_profile(llm, cols)


def run(df: pd.DataFrame, title: str, llm, *, crosscheck: bool = False,
        defenses: tuple[str, ...] | list[str] = (), profile: dict | None = None,
        semantic: dict | None = None) -> Result:
    defenses = set(defenses)
    report: dict = {}
    view = df

    if "sanitize" in defenses:
        view, report["sanitize"] = D.sanitize(view)

    untrusted: set[str] = set()
    if "stats_only" in defenses:
        untrusted = D.untrusted_columns(view)
        report["untrusted_columns"] = sorted(untrusted)

    if "consistency" in defenses:
        trusted = view.drop(columns=list(D.untrusted_columns(view) | D.suspicious_names(view)))
        t_full = G.topic(llm, title, G.dataset_sample_text(_llm_sample(view, untrusted)))
        t_trusted = G.topic(llm, title, G.dataset_sample_text(sample_rows(trusted)))
        ok = D.topics_agree(t_full, t_trusted)
        report["consistency"] = {"topic_full": t_full, "topic_trusted": t_trusted, "agree": ok}
        if not ok:
            view, untrusted = trusted, set()

    prof = profile if profile is not None else profile_df(view, title)
    if untrusted:
        prof = {**prof, "columns": [{**c, "top_values": []} if c["name"] in untrusted else c
                                    for c in prof["columns"]]}
    sem = semantic if semantic is not None else semantic_for(llm, view, untrusted)

    flags = run_crosscheck(prof, sem) if crosscheck else []
    profile_text = render_profile(prof, withheld_map(flags))
    semantic_text = G.render_semantic(sem)

    sample_text = G.dataset_sample_text(_llm_sample(view, untrusted))
    topic = G.topic(llm, title, sample_text)
    ufd = G.ufd(llm, sample_text, profile_text, semantic_text, topic)
    sfd = G.sfd(llm, ufd, topic)
    return Result(title, topic, ufd, sfd, profile_text, semantic_text, sem,
                  [f.dict() for f in flags], report)


def _llm_sample(view: pd.DataFrame, untrusted: set[str]) -> pd.DataFrame:
    return sample_rows(view.drop(columns=[c for c in view.columns if str(c) in untrusted]))
