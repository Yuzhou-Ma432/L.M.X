#!/usr/bin/env python3
"""Command-line entry point for the L.M.X pipeline.

Usage:
    python main.py collect                # select datasets from NYC Open Data
    python main.py profile  [--limit N]   # profile the selected datasets
    python main.py generate [--limit N]   # generate descriptions (3 strategies)
    python main.py evaluate               # compare cost/time/similarity
    python main.py pipeline [--limit N]   # collect + profile + generate + evaluate

`--limit N` processes only the first N selected datasets (handy for smoke tests).
"""
from __future__ import annotations

import argparse
import json


def _load_selected() -> list[dict]:
    return json.loads((__import__("config").CATALOG_DIR / "selected.json")
                      .read_text(encoding="utf-8"))


def main() -> None:
    import config
    from src import collect, evaluate, generate, profile

    p = argparse.ArgumentParser(description="L.M.X dataset-description pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("collect")
    for name in ("profile", "generate", "pipeline"):
        sp = sub.add_parser(name)
        sp.add_argument("--limit", type=int, default=None)
        if name in ("generate", "pipeline"):
            sp.add_argument("--workers", type=int, default=1,
                            help="concurrent LLM calls (e.g. 6)")
    sp = sub.add_parser("evaluate")

    args = p.parse_args()

    if args.cmd == "collect":
        collect.collect()

    elif args.cmd == "profile":
        profile.profile_all(_load_selected(), limit=args.limit)

    elif args.cmd == "generate":
        profiles = [json.loads(f.read_text(encoding="utf-8"))
                    for f in sorted(config.PROFILE_DIR.glob("*.json"))]
        generate.generate_all(profiles[: args.limit] if args.limit else profiles,
                              workers=args.workers)

    elif args.cmd == "evaluate":
        results = evaluate.evaluate()
        print(evaluate.summarize(results))

    elif args.cmd == "pipeline":
        collect.collect()
        profile.profile_all(_load_selected(), limit=args.limit)
        profiles = [json.loads(f.read_text(encoding="utf-8"))
                    for f in sorted(config.PROFILE_DIR.glob("*.json"))]
        generate.generate_all(profiles[: args.limit] if args.limit else profiles,
                              workers=args.workers)
        results = evaluate.evaluate()
        print(evaluate.summarize(results))


if __name__ == "__main__":
    main()
