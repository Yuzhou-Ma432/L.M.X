"""Experiment 1 — profiler-error propagation and the cross-check stage.

For every dataset and every applicable error operator:

  clean      profile as computed
  corrupt    one derived statistic corrupted (inject_errors)
  corrupt+cc corrupted profile, but the cross-check withholds disputed facts

Measured:
  * propagation  — fake signal appears in the description (and not in the clean one)
  * true loss    — a true fact present in the clean description disappears
  * detection    — the cross-check flags the corrupted column/field (recall),
                   and flags on clean profiles (false-positive rate)
  * retrieval    — reciprocal rank of the dataset for (a) its known-item query
                   and (b) for temporal errors, that query + its true years,
                   with only this dataset's SFD swapped in a BM25 index

Usage:  python -m experiments.exp1_propagation [--llm openai] [--limit 10]
"""
from __future__ import annotations

import json
from collections import defaultdict

from robustddg import corpus as C
from robustddg.inject_errors import OPS, inject, present
from robustddg.pipeline import run, semantic_for
from robustddg.profiler import profile_df
from robustddg.search import BM25, rr

from .common import base_parser, make_llm, md_table, mean, out_dir, write_jsonl


def main() -> None:
    args = base_parser(__doc__.splitlines()[1]).parse_args()
    llm = make_llm(args)
    out = out_dir(args, "exp1", llm)
    data = C.load(args.corpus)[: args.limit]
    queries = C.known_item_queries(data)
    print(f"{len(data)} datasets, llm={llm.model}")

    clean_sfd: dict[str, str] = {}
    rows: list[dict] = []
    fp_rows: list[dict] = []
    for i, d in enumerate(data, 1):
        df, title = d["df"], d["title"]
        prof = profile_df(df, title)
        sem = semantic_for(llm, df)
        clean = run(df, title, llm, profile=prof, semantic=sem)
        clean_cc = run(df, title, llm, profile=prof, semantic=sem, crosscheck=True)
        clean_sfd[d["id"]] = clean.sfd
        fp_rows.append({"id": d["id"], "n_cols": prof["n_cols"], "flags": clean_cc.flags,
                        "sfd_cc": clean_cc.sfd})
        for op in OPS:
            inj = inject(prof, op, d["id"])
            if inj is None:
                continue
            bad_prof, rec = inj
            bad = run(df, title, llm, profile=bad_prof, semantic=sem)
            bad_cc = run(df, title, llm, profile=bad_prof, semantic=sem, crosscheck=True)
            col_flags = [f for f in bad_cc.flags if f["column"] == rec.column]
            row = {"id": d["id"], "title": title, **rec.dict(),
                   "detected": bool(col_flags),
                   "detected_field": any(f["field"] == rec.field for f in col_flags),
                   "rules": sorted({f["rule"] for f in col_flags}),
                   "n_flags_other_columns": len(bad_cc.flags) - len(col_flags)}
            for cond, res in (("clean", clean), ("corrupt", bad), ("corrupt_cc", bad_cc)):
                for kind in ("ufd", "sfd"):
                    txt = getattr(res, kind)
                    row[f"{cond}_{kind}_fake"] = present(txt, rec.fake)
                    row[f"{cond}_{kind}_true"] = present(txt, rec.true)
                row[f"{cond}_sfd"] = res.sfd
                row[f"{cond}_ufd"] = res.ufd
            rows.append(row)
        print(f"[{i}/{len(data)}] {d['id']} {title[:50]}")

    # ---- retrieval: swap one SFD at a time -----------------------------------
    base = BM25(clean_sfd)
    for r in rows:
        q = queries[r["id"]]
        qs = {"known_item": q}
        if r["op"] == "date_range" and r["true"].get("years"):
            qs["fact"] = q + " " + " ".join(map(str, r["true"]["years"]))
        for qname, qtext in qs.items():
            r[f"rr_clean_{qname}"] = rr(base.rank_of(r["id"], qtext))
            for cond in ("corrupt", "corrupt_cc"):
                idx = BM25({**clean_sfd, r["id"]: r[f"{cond}_sfd"]})
                r[f"rr_{cond}_{qname}"] = rr(idx.rank_of(r["id"], qtext))

    write_jsonl(out / "records.jsonl", rows)
    write_jsonl(out / "clean_crosscheck.jsonl", fp_rows)
    summary = summarize(rows, fp_rows, llm)
    (out / "summary.md").write_text(summary)
    print(summary)
    print(f"\nwrote {out}/")


def _propagated(r: dict, cond: str, kind: str) -> bool:
    return r[f"{cond}_{kind}_fake"] and not r[f"clean_{kind}_fake"]


def _lost(r: dict, cond: str, kind: str) -> bool:
    return r[f"clean_{kind}_true"] and not r[f"{cond}_{kind}_true"]


def summarize(rows: list[dict], fp_rows: list[dict], llm) -> str:
    by_op = defaultdict(list)
    for r in rows:
        by_op[r["op"]].append(r)
    by_op["ALL"] = rows

    prop = []
    for op, rs in by_op.items():
        prop.append([op, len(rs)] + [mean([_propagated(r, c, k) for r in rs])
                                     for k in ("ufd", "sfd") for c in ("corrupt", "corrupt_cc")]
                    + [mean([_lost(r, c, "ufd") for r in rs]) for c in ("corrupt", "corrupt_cc")])
    det = [[op, len(rs), mean([r["detected"] for r in rs]), mean([r["detected_field"] for r in rs]),
            ", ".join(sorted({x for r in rs for x in r["rules"]}))] for op, rs in by_op.items()]
    n_cols = sum(f["n_cols"] for f in fp_rows)
    n_flag_cols = sum(len({x["column"] for x in f["flags"]}) for f in fp_rows)
    rules = defaultdict(int)
    for f in fp_rows:
        for x in f["flags"]:
            rules[x["rule"]] += 1

    ret = []
    for op, rs in by_op.items():
        for q in ("known_item", "fact"):
            have = [r for r in rs if f"rr_clean_{q}" in r]
            if have:
                ret.append([op, q, len(have)] + [mean([r[f"rr_{c}_{q}"] for r in have])
                                                for c in ("clean", "corrupt", "corrupt_cc")])

    mock_note = ("\n> **MockLLM run** — the mock copies every profile fact verbatim, so "
                 "propagation here is an upper bound that validates plumbing only.\n"
                 if llm.model == "mock" else "")
    return "\n".join([
        f"# Exp 1 — profiler-error propagation (llm={llm.model})", mock_note,
        "## Propagation (fraction of injected errors whose fake signal reaches the text)",
        md_table(["op", "n", "UFD", "UFD +cc", "SFD", "SFD +cc", "true-fact loss UFD",
                  "loss UFD +cc"], prop),
        "", "## Cross-check detection on corrupted profiles",
        md_table(["op", "n", "column flagged", "right field flagged", "rules fired"], det),
        "", "## Cross-check on clean profiles (false positives)",
        f"- datasets with >=1 flag: {mean([bool(f['flags']) for f in fp_rows]):.3f} "
        f"({sum(bool(f['flags']) for f in fp_rows)}/{len(fp_rows)})",
        f"- columns flagged: {n_flag_cols}/{n_cols} = {n_flag_cols / max(n_cols, 1):.3f}",
        f"- by rule: {json.dumps(dict(rules))}",
        "", "## Retrieval (MRR of the affected dataset, only its SFD swapped)",
        md_table(["op", "query", "n", "clean", "corrupt", "corrupt +cc"], ret),
    ])


if __name__ == "__main__":
    main()
