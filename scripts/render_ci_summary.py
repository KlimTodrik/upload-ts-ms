#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GitHub Actions summary from benchmark JSON.")
    parser.add_argument("--result", required=True)
    parser.add_argument("--summary-file", required=True)
    return parser.parse_args()


def badge(label: str, value: str, color: str) -> str:
    return f"![{label}](https://img.shields.io/badge/{quote(label)}-{quote(value)}-{color})"


def format_engine_row(name: str, payload: dict) -> str:
    seconds = float(payload["seconds"])
    schema_setup_seconds = float(payload.get("schema_setup_seconds", 0.0))
    docs_per_second = float(payload["docs_per_second"])
    return (
        f"| {name} | {seconds:.4f} | {schema_setup_seconds:.4f} | "
        f"{docs_per_second:.2f} | {int(payload['rows'])} | {int(payload['batch_size'])} |"
    )


def main() -> int:
    args = parse_args()
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))

    manticore = result.get("manticore")
    typesense = result.get("typesense")
    if not manticore or not typesense:
        raise SystemExit("Both manticore and typesense results are required for CI summary")

    winner = "Manticore" if manticore["seconds"] < typesense["seconds"] else "Typesense"
    summary_lines = [
        "# Upload Benchmark",
        "",
        "## Snapshot",
        "",
        f"- Dataset: `{result['dataset']}`",
        f"- Rows: `{result['rows']}`",
        f"- Mode: `{result['mode']}`",
        f"- Embedding model: `{result.get('embedding_model') or 'engine-native auto mode'}`",
        f"- Embedding generation: `{float(result['embedding_generation_seconds']):.4f}s`",
        "- `Seconds` below means import-only time. Schema setup is shown separately.",
        "",
        "## Badges",
        "",
        badge("Winner", winner, "brightgreen"),
        "",
        badge("Manticore seconds", f"{float(manticore['seconds']):.4f}s", "1f6feb"),
        " ",
        badge("Typesense seconds", f"{float(typesense['seconds']):.4f}s", "0a7f5a"),
        "",
        badge("Manticore docs/sec", f"{float(manticore['docs_per_second']):.2f}", "1f6feb"),
        " ",
        badge("Typesense docs/sec", f"{float(typesense['docs_per_second']):.2f}", "0a7f5a"),
        "",
        "## Table",
        "",
        "| Engine | Import seconds | Schema setup seconds | Docs/sec | Rows | Batch size |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        format_engine_row("Manticore", manticore),
        format_engine_row("Typesense", typesense),
        "",
        f"Raw result: `{args.result}`",
        "",
    ]

    summary_text = "\n".join(summary_lines)
    Path(args.summary_file).write_text(summary_text, encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
