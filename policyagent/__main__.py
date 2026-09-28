"""Command-line interface.

    python -m policyagent ask "How long must case records be kept?"
    python -m policyagent eval [--out outputs/eval.json]
    python -m policyagent build-index
    python -m policyagent serve [--port 8000]

Add --offline before the command to force local providers even if Azure is configured.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)  # real environment variables win over .env


def _print_answer(response) -> None:
    print(f"[{response.outcome}] (mode: {response.mode}{', degraded' if response.degraded else ''})")
    print(response.answer)
    for c in response.citations:
        print(f"  [{c.number}] {c.title} > {c.section} ({c.source})")
    if response.reasons:
        print(f"  reasons: {', '.join(response.reasons)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="policyagent", description="Grounded government policy Q&A agent")
    parser.add_argument("--offline", action="store_true", help="force offline providers")
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ev = sub.add_parser("eval", help="run the evaluation set")
    ev.add_argument("--out", type=Path, help="write the full report to this JSON file")
    sub.add_parser("build-index", help="create the Azure AI Search index and upload chunks")
    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    _load_dotenv()
    settings = Settings.from_env(force_offline=args.offline)

    if args.command == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(settings=settings), host=args.host, port=args.port)
        return 0
    if args.command == "build-index":
        from .indexer import build_index

        print(f"Indexed {build_index(settings)} chunks into '{settings.search_index}'.")
        return 0

    from .agent import build_agent
    from .telemetry import setup_tracing

    setup_tracing(settings)
    agent = build_agent(settings)
    if args.command == "ask":
        response = agent.ask(args.question)
        print(json.dumps(response.to_dict(), indent=2)) if args.json else _print_answer(response)
        return 0

    from .evaluation import run_evaluation

    report = run_evaluation(agent, settings)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"mode: {report['mode']}  cases: {report['cases']}")
    for name, value in {**report["offline"], **report.get("azure_ai_evaluation", {})}.items():
        print(f"  {name:<20} {value}")
    failures = [r["id"] for r in report["rows"] if False in (r.get("abstention_correct"), r.get("safety_correct"))]
    if failures:
        print(f"  failing cases: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
