#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


RESULTS_DIR = Path("results")
CACHE_DIR = Path(".cache")
DATASET_NAME = "BeIR/fiqa"
PRECOMPUTED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MANTICORE_AUTO_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TYPESENSE_AUTO_MODEL = "ts/all-MiniLM-L12-v2"
DEFAULT_COLLECTION = "fiqa_bench"
BEIR_FIQA_PARQUET_URL = (
    "https://huggingface.co/datasets/BeIR/fiqa/resolve/"
    "ecb5eb6dcbf64d9eb5b9b48ef4fcd925af0ea056/corpus/fiqa-corpus.parquet"
)


@dataclass
class EngineResult:
    seconds: float
    docs_per_second: float
    rows: int
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark upload speed for Manticore and Typesense.")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=("precomputed", "auto"),
        default="precomputed",
        help="precomputed = same external embeddings for both engines; auto = each engine embeds on write",
    )
    parser.add_argument(
        "--engines",
        choices=("manticore", "typesense", "both"),
        default="both",
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--manticore-host", default="127.0.0.1")
    parser.add_argument("--manticore-port", type=int, default=19306)
    parser.add_argument("--typesense-url", default="http://127.0.0.1:18108")
    parser.add_argument("--typesense-api-key", default="xyz")
    parser.add_argument("--embed-model", default=PRECOMPUTED_MODEL)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def require_requests():
    import requests  # type: ignore

    return requests


def require_datasets():
    from datasets import load_dataset  # type: ignore

    return load_dataset


def require_sentence_transformer():
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer


def require_pymysql():
    import pymysql  # type: ignore

    return pymysql


def chunked(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def load_fiqa(dataset_name: str, limit: int) -> list[dict[str, Any]]:
    load_dataset = require_datasets()
    try:
        ds = load_dataset(dataset_name, "corpus", split="corpus")
    except RuntimeError as exc:
        if dataset_name != DATASET_NAME or "Dataset scripts are no longer supported" not in str(exc):
            raise
        parquet_path = ensure_fiqa_parquet()
        ds = load_dataset("parquet", data_files={"corpus": str(parquet_path)}, split="corpus")

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(ds):
        if idx >= limit:
            break
        title = (item.get("title") or "").strip()
        text = (item.get("text") or "").strip()
        rows.append(
            {
                "id": idx + 1,
                "source_id": str(item.get("_id", idx + 1)),
                "title": title,
                "description": text,
                "embedding_input": " ".join(part for part in (title, text) if part).strip(),
            }
        )
    return rows


def add_precomputed_embeddings(rows: list[dict[str, Any]], model_name: str, batch_size: int) -> float:
    SentenceTransformer = require_sentence_transformer()
    model = SentenceTransformer(model_name)
    texts = [row["embedding_input"] for row in rows]

    started = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    elapsed = time.perf_counter() - started

    for row, vector in zip(rows, vectors):
        row["embedding"] = [float(x) for x in vector.tolist()]
    return elapsed


class ManticoreClient:
    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.pymysql = require_pymysql()
        self.host = host
        self.port = port
        self.timeout = timeout
        self.conn = None

    def connect(self):
        if self.conn is None or not self.conn.open:
            self.conn = self.pymysql.connect(
                host=self.host,
                port=self.port,
                user="",
                password="",
                autocommit=True,
                charset="utf8mb4",
                connect_timeout=int(self.timeout),
                read_timeout=int(self.timeout),
                write_timeout=int(self.timeout),
            )
        return self.conn

    def wait_ready(self) -> None:
        deadline = time.time() + self.timeout
        last_error = None
        while time.time() < deadline:
            try:
                conn = self.connect()
                with conn.cursor() as cursor:
                    cursor.execute("SHOW TABLES")
                    cursor.fetchall()
                return
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
                self.conn = None
            time.sleep(1)
        raise RuntimeError(f"Manticore SQL is not ready: {last_error}")

    def sql(self, statement: str) -> Any:
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(statement)
            return cursor.fetchall()

    def recreate_table(self, table_name: str, mode: str, num_dim: int | None) -> None:
        self.sql(f"DROP TABLE IF EXISTS {table_name}")
        if mode == "precomputed":
            if not num_dim:
                raise ValueError("num_dim is required for manual Manticore vectors")
            ddl = f"""
            CREATE TABLE {table_name} (
                id BIGINT,
                source_id TEXT,
                title TEXT,
                description TEXT,
                embedding FLOAT_VECTOR KNN_TYPE='hnsw' KNN_DIMS='{num_dim}' HNSW_SIMILARITY='cosine'
            )
            """
        else:
            ddl = f"""
            CREATE TABLE {table_name} (
                id BIGINT,
                source_id TEXT,
                title TEXT,
                description TEXT,
                embedding FLOAT_VECTOR
                    KNN_TYPE='hnsw'
                    HNSW_SIMILARITY='cosine'
                    MODEL_NAME='{MANTICORE_AUTO_MODEL}'
                    FROM='title,description'
            )
            """
        self.sql(" ".join(ddl.split()))

    def import_rows(self, table_name: str, rows: list[dict[str, Any]], batch_size: int, mode: str) -> EngineResult:
        conn = self.connect()
        started = time.perf_counter()
        for batch in chunked(rows, batch_size):
            values_sql = []
            for row in batch:
                source_id = conn.escape(row["source_id"])
                title = conn.escape(row["title"])
                description = conn.escape(row["description"])
                if mode == "precomputed":
                    vector_sql = "(" + ",".join(f"{float(x):.8f}" for x in row["embedding"]) + ")"
                    values_sql.append(
                        f"({row['id']}, {source_id}, {title}, {description}, {vector_sql})"
                    )
                else:
                    values_sql.append(f"({row['id']}, {source_id}, {title}, {description})")

            if mode == "precomputed":
                sql = (
                    f"INSERT INTO {table_name} "
                    f"(id, source_id, title, description, embedding) VALUES "
                    + ", ".join(values_sql)
                )
            else:
                sql = (
                    f"INSERT INTO {table_name} "
                    f"(id, source_id, title, description) VALUES "
                    + ", ".join(values_sql)
                )
            self.sql(sql)

        elapsed = time.perf_counter() - started
        return EngineResult(
            seconds=elapsed,
            docs_per_second=(len(rows) / elapsed) if elapsed > 0 else 0.0,
            rows=len(rows),
            batch_size=batch_size,
        )


class TypesenseClient:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self.requests = require_requests()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"X-TYPESENSE-API-KEY": api_key}

    def wait_ready(self) -> None:
        deadline = time.time() + self.timeout
        last_error = None
        while time.time() < deadline:
            try:
                response = self.requests.get(
                    f"{self.base_url}/health",
                    headers=self.headers,
                    timeout=2,
                )
                if response.ok:
                    return
                last_error = response.text
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
            time.sleep(1)
        raise RuntimeError(f"Typesense is not ready: {last_error}")

    def recreate_collection(self, collection_name: str, mode: str, num_dim: int | None) -> None:
        self.requests.delete(
            f"{self.base_url}/collections/{collection_name}",
            headers=self.headers,
            timeout=self.timeout,
        )

        fields: list[dict[str, Any]] = [
            {"name": "id", "type": "string"},
            {"name": "source_id", "type": "string"},
            {"name": "title", "type": "string"},
            {"name": "description", "type": "string"},
        ]

        if mode == "precomputed":
            if not num_dim:
                raise ValueError("num_dim is required for precomputed Typesense schema")
            fields.append({"name": "embedding", "type": "float[]", "num_dim": num_dim})
        else:
            fields.append(
                {
                    "name": "embedding",
                    "type": "float[]",
                    "embed": {
                        "from": ["title", "description"],
                        "model_config": {"model_name": TYPESENSE_AUTO_MODEL},
                    },
                }
            )

        schema = {"name": collection_name, "fields": fields}
        response = self.requests.post(
            f"{self.base_url}/collections",
            headers={**self.headers, "Content-Type": "application/json"},
            json=schema,
            timeout=max(self.timeout, 120),
        )
        response.raise_for_status()

    def import_rows(self, collection_name: str, rows: list[dict[str, Any]], batch_size: int, mode: str) -> EngineResult:
        started = time.perf_counter()
        for batch in chunked(rows, batch_size):
            lines = []
            for row in batch:
                doc = {
                    "id": str(row["id"]),
                    "source_id": row["source_id"],
                    "title": row["title"],
                    "description": row["description"],
                }
                if mode == "precomputed":
                    doc["embedding"] = row["embedding"]
                lines.append(json.dumps(doc))

            response = self.requests.post(
                f"{self.base_url}/collections/{collection_name}/documents/import",
                params={"action": "upsert"},
                headers={**self.headers, "Content-Type": "text/plain"},
                data="\n".join(lines) + "\n",
                timeout=max(self.timeout, 120),
            )
            response.raise_for_status()
            if '"success": false' in response.text:
                raise RuntimeError(f"Typesense import error: {response.text}")

        elapsed = time.perf_counter() - started
        return EngineResult(
            seconds=elapsed,
            docs_per_second=(len(rows) / elapsed) if elapsed > 0 else 0.0,
            rows=len(rows),
            batch_size=batch_size,
        )


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)


def ensure_fiqa_parquet() -> Path:
    requests = require_requests()
    dataset_cache_dir = CACHE_DIR / "datasets"
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = dataset_cache_dir / "fiqa-corpus.parquet"
    if parquet_path.exists() and parquet_path.stat().st_size > 0:
        return parquet_path

    response = requests.get(BEIR_FIQA_PARQUET_URL, stream=True, timeout=120)
    response.raise_for_status()
    with parquet_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return parquet_path


def write_results(payload: dict[str, Any], mode: str, limit: int) -> Path:
    ensure_results_dir()
    ts = time.strftime("%Y%m%dT%H%M%S")
    path = RESULTS_DIR / f"benchmark-{mode}-{limit}-{ts}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()

    rows = load_fiqa(args.dataset, args.limit)
    if not rows:
        raise RuntimeError("Dataset is empty")

    embedding_generation_seconds = 0.0
    if args.mode == "precomputed":
        embedding_generation_seconds = add_precomputed_embeddings(
            rows,
            model_name=args.embed_model,
            batch_size=args.embed_batch_size,
        )
        num_dim = len(rows[0]["embedding"])
    else:
        num_dim = None

    output: dict[str, Any] = {
        "dataset": args.dataset,
        "rows": len(rows),
        "mode": args.mode,
        "embedding_model": args.embed_model if args.mode == "precomputed" else None,
        "embedding_generation_seconds": round(embedding_generation_seconds, 4),
        "notes": {
            "precomputed": "same external embeddings for both engines",
            "auto": {
                "manticore_model": MANTICORE_AUTO_MODEL,
                "typesense_model": TYPESENSE_AUTO_MODEL,
            },
        },
    }

    if args.engines in ("manticore", "both"):
        client = ManticoreClient(args.manticore_host, args.manticore_port, timeout=args.timeout)
        client.wait_ready()
        client.recreate_table(args.collection, mode=args.mode, num_dim=num_dim)
        result = client.import_rows(args.collection, rows, args.batch_size, mode=args.mode)
        output["manticore"] = asdict(result)

    if args.engines in ("typesense", "both"):
        client = TypesenseClient(args.typesense_url, args.typesense_api_key, timeout=args.timeout)
        client.wait_ready()
        client.recreate_collection(args.collection, mode=args.mode, num_dim=num_dim)
        result = client.import_rows(args.collection, rows, args.batch_size, mode=args.mode)
        output["typesense"] = asdict(result)

    result_path = write_results(output, args.mode, len(rows))
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {result_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
