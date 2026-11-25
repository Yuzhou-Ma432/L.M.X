"""Shared CLI plumbing for the experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--llm", choices=["mock", "openai"], default="mock",
                   help="'openai' = any OpenAI-compatible endpoint (LLM_BASE_URL/LLM_API_KEY/LLM_MODEL)")
    p.add_argument("--model", default=None)
    p.add_argument("--limit", type=int, default=None, help="use only the first N datasets")
    p.add_argument("--out", default=None)
    return p


def make_llm(args):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if args.llm == "mock":
        from robustddg.llm import MockLLM
        return MockLLM()
    from robustddg.llm import OpenAICompatClient
    if not os.getenv("LLM_API_KEY"):
        raise SystemExit("LLM_API_KEY is not set (see .env.example)")
    return OpenAICompatClient(model=args.model)


def out_dir(args, name: str, llm) -> Path:
    d = Path(args.out or f"results/{name}_{getattr(llm, 'model', 'llm').replace('/', '_')}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def md_table(headers: list[str], rows: list[list]) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")
