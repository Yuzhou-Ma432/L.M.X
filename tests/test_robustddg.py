import numpy as np
import pandas as pd
import pytest

from robustddg import attacks, defenses
from robustddg.crosscheck import crosscheck
from robustddg.inject_errors import OPS, inject, numbers_in, present, years_in
from robustddg.llm import MockLLM
from robustddg.pipeline import run, semantic_for
from robustddg.profiler import profile_df
from robustddg.search import BM25


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "crash_date": pd.date_range("2019-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "year": rng.integers(2015, 2023, n),
        "borough": rng.choice(["BRONX", "BROOKLYN", "QUEENS"], n),
        "number_of_persons_injured": rng.integers(0, 6, n),
        "latitude": rng.uniform(40.5, 40.9, n).round(5),
        "total_cost_usd": rng.uniform(100, 5000, n).round(2),
    })


def test_profile_types(df):
    p = profile_df(df, "Collisions")
    t = {c["name"]: c["type"] for c in p["columns"]}
    assert t["crash_date"] == "datetime" and t["year"] == "integer"
    assert t["borough"] == "categorical" and t["latitude"] == "float"
    units = {c["name"]: c["unit"] for c in p["columns"]}
    assert units["total_cost_usd"] == "US dollars" and units["number_of_persons_injured"] == "count"


def test_clean_profile_has_no_flags(df):
    p = profile_df(df, "Collisions")
    sem = semantic_for(MockLLM(), df)
    assert crosscheck(p, sem) == []


@pytest.mark.parametrize("op", OPS)
def test_every_injection_is_detected(df, op):
    p = profile_df(df, "Collisions")
    sem = semantic_for(MockLLM(), df)
    out = inject(p, op, "seed")
    assert out is not None
    bad, rec = out
    flagged = {(f.column, f.field) for f in crosscheck(bad, sem)}
    assert (rec.column, rec.field) in flagged, (rec, flagged)


def test_crosscheck_blocks_propagation(df):
    llm = MockLLM()
    p = profile_df(df, "Collisions")
    sem = semantic_for(llm, df)
    bad, rec = inject(p, "date_range", "seed")
    plain = run(df, "Collisions", llm, profile=bad, semantic=sem)
    gated = run(df, "Collisions", llm, profile=bad, semantic=sem, crosscheck=True)
    assert present(plain.ufd, rec.fake) and not present(gated.ufd, rec.fake)


def test_probes():
    assert years_in("from 1982 to 2019, id 12345") == {1982, 2019}
    assert any(abs(x - 4.5e6) < 1 for x in numbers_in("up to 4.5 million dollars"))
    assert present("costs $5", {"terms": [r"\$"]})


def test_injection_hijacks_mock_and_defenses_stop_it(df):
    llm = MockLLM()
    for name in ("colname_injection", "cell_injection"):
        poisoned = attacks.ATTACKS[name](df)
        assert "covid" in run(poisoned, "Collisions", llm).ufd.lower()
        assert "covid" not in run(poisoned, "Collisions", llm, defenses=("sanitize",)).ufd.lower()
    # stats-only removes the cell payload but cannot fix a poisoned column *name*
    assert "covid" not in run(attacks.cell_injection(df), "Collisions", llm,
                              defenses=("stats_only",)).ufd.lower()
    r = run(attacks.colname_injection(df), "Collisions", llm, defenses=("consistency",))
    assert not r.defense_report["consistency"]["agree"] and "covid" not in r.ufd.lower()


def test_seo_stuffing_detected():
    assert defenses.is_keyword_list(attacks.SEO_KEYWORDS)
    assert not defenses.is_keyword_list("Bronx, NY 10451")
    assert defenses.is_instruction(attacks.INSTRUCTION)
    assert not defenses.is_instruction("Vehicle was parked on the left side")


def test_bm25_ranks_relevant_first():
    idx = BM25({"a": "covid vaccine doses by zip code", "b": "motor vehicle collisions",
                "c": "restaurant inspections"})
    assert idx.ranking("covid-19 vaccination")[0] == "a"
    assert idx.rank_of("b", "covid vaccine") is None
