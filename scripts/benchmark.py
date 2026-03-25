#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


RESULTS_DIR = Path("results")
CACHE_DIR = Path(".cache")
DATASET_NAME = "BeIR/fiqa"
PRECOMPUTED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MANTICORE_AUTO_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
TYPESENSE_AUTO_MODEL = "ts/all-MiniLM-L12-v2"
DEFAULT_COLLECTION = "fiqa_bench"
DEFAULT_MANTICORE_CONTAINER = "upload-compare-manticore"
DEFAULT_TYPESENSE_CONTAINER = "upload-compare-typesense"
BEIR_FIQA_PARQUET_URL = (
    "https://huggingface.co/datasets/BeIR/fiqa/resolve/"
    "ecb5eb6dcbf64d9eb5b9b48ef4fcd925af0ea056/corpus/fiqa-corpus.parquet"
)


@dataclass
class EngineResult:
    seconds: float
    schema_setup_seconds: float
    docs_per_second: float
    rows: int
    batch_size: int
    resource_usage: dict[str, Any] | None = None


def log(message: str) -> None:
    print(message, flush=True)


def schema_timeout_for_mode(mode: str, base_timeout: float) -> float:
    return max(base_timeout, 600.0) if mode == "auto" else base_timeout


def should_log_batch_progress(batch_idx: int, total_batches: int) -> bool:
    return batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0


class StageTimer:
    def __init__(self, label: str) -> None:
        self.label = label
        self.started = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        log(f"[stage] {self.label} started")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.perf_counter() - self.started
        if exc_type is None:
            log(f"[stage] {self.label} finished in {elapsed:.4f}s")
        else:
            log(f"[stage] {self.label} failed after {elapsed:.4f}s")


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
    parser.add_argument("--manticore-container", default=DEFAULT_MANTICORE_CONTAINER)
    parser.add_argument("--typesense-container", default=DEFAULT_TYPESENSE_CONTAINER)
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=0.5,
        help="docker stats polling interval in seconds for CPU/memory sampling",
    )
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


BYTE_UNITS = {
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mb": 1000**2,
    "mib": 1024**2,
    "gb": 1000**3,
    "gib": 1024**3,
    "tb": 1000**4,
    "tib": 1024**4,
}


def parse_size_bytes(raw: str) -> int:
    value = raw.strip()
    if not value:
        raise ValueError("size value is empty")

    match = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)?\s*$", value)
    if not match:
        raise ValueError(f"unsupported size format: {raw}")
    number = float(match.group(1))
    unit = (match.group(2) or "B").lower()
    multiplier = BYTE_UNITS.get(unit)
    if multiplier is None:
        raise ValueError(f"unsupported size unit: {unit}")
    return int(number * multiplier)


def parse_percent(raw: str) -> float:
    return float(raw.strip().rstrip("%"))


def format_bytes(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


def build_resource_usage_summary(samples: list[dict[str, float | int]]) -> dict[str, Any]:
    memory_limit_values = [int(sample["memory_limit_bytes"]) for sample in samples if int(sample["memory_limit_bytes"]) > 0]
    memory_limit_bytes = max(memory_limit_values) if memory_limit_values else 0
    cpu_values = [float(sample["cpu_percent"]) for sample in samples]
    memory_values = [int(sample["memory_used_bytes"]) for sample in samples]
    interval_values = [
        max(float(samples[i]["elapsed_seconds"]) - float(samples[i - 1]["elapsed_seconds"]), 0.0)
        for i in range(1, len(samples))
    ]
    summary: dict[str, Any] = {
        "sample_count": len(samples),
        "sampling_interval_seconds": round(sum(interval_values) / len(interval_values), 4) if interval_values else 0.0,
        "cpu_percent_avg": round(sum(cpu_values) / len(cpu_values), 2),
        "cpu_percent_peak": round(max(cpu_values), 2),
        "memory_used_bytes_avg": int(sum(memory_values) / len(memory_values)),
        "memory_used_bytes_peak": max(memory_values),
    }
    if memory_limit_bytes > 0:
        summary["memory_limit_bytes"] = memory_limit_bytes
        summary["memory_percent_peak"] = round(max(memory_values) / memory_limit_bytes * 100, 2)
    return summary


def format_resource_usage(resource_usage: dict[str, Any] | None) -> str:
    if not resource_usage:
        return "resource usage unavailable"
    if resource_usage.get("error"):
        return (
            "resource usage unavailable: "
            f"{resource_usage['error']}"
        )

    cpu_avg = resource_usage.get("cpu_percent_avg")
    cpu_peak = resource_usage.get("cpu_percent_peak")
    mem_avg = resource_usage.get("memory_used_bytes_avg")
    mem_peak = resource_usage.get("memory_used_bytes_peak")
    memory_peak_percent = resource_usage.get("memory_percent_peak")

    memory_peak_suffix = ""
    if memory_peak_percent is not None:
        memory_peak_suffix = f" ({float(memory_peak_percent):.2f}% limit)"

    return (
        f"cpu avg={float(cpu_avg):.2f}% peak={float(cpu_peak):.2f}% | "
        f"mem avg={format_bytes(mem_avg)} peak={format_bytes(mem_peak)}{memory_peak_suffix}"
    )


class DockerStatsMonitor:
    def __init__(self, container_name: str, interval_seconds: float) -> None:
        self.container_name = container_name
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float | int]] = []
        self.error: str | None = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def start(self) -> None:
        if shutil.which("docker") is None:
            self.error = "docker CLI is not available"
            return
        self._started_at = time.perf_counter()
        self._running.set()
        self._thread = threading.Thread(target=self._run, name=f"docker-stats-{self.container_name}", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any] | None:
        if self._thread is None:
            return None
        self._running.clear()
        self._thread.join(timeout=max(self.interval_seconds * 4, 2.0))
        if self.error:
            return {"container": self.container_name, "error": self.error}
        if not self.samples:
            return {"container": self.container_name, "error": "no docker stats samples collected"}
        summary = build_resource_usage_summary(self.samples)
        summary["container"] = self.container_name
        return summary

    def _run(self) -> None:
        while self._running.is_set():
            try:
                sample = self._collect_sample()
                if sample is not None:
                    self.samples.append(sample)
            except Exception as exc:
                self.error = str(exc)
                self._running.clear()
                return
            time.sleep(self.interval_seconds)

    def _collect_sample(self) -> dict[str, float | int] | None:
        command = [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{.CPUPerc}}\t{{.MemUsage}}",
            self.container_name,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(stderr or f"docker stats failed for {self.container_name}")

        line = completed.stdout.strip()
        if not line:
            return None
        cpu_raw, mem_raw = line.split("\t", 1)
        memory_used_raw, _, memory_limit_raw = mem_raw.partition("/")
        return {
            "elapsed_seconds": round(time.perf_counter() - self._started_at, 4),
            "cpu_percent": parse_percent(cpu_raw),
            "memory_used_bytes": parse_size_bytes(memory_used_raw),
            "memory_limit_bytes": parse_size_bytes(memory_limit_raw) if memory_limit_raw.strip() else 0,
        }


def load_fiqa(dataset_name: str, limit: int) -> list[dict[str, Any]]:
    log(f"[dataset] loading dataset={dataset_name} limit={limit}")
    load_dataset = require_datasets()
    try:
        ds = load_dataset(dataset_name, "corpus", split="corpus")
    except RuntimeError as exc:
        if dataset_name != DATASET_NAME or "Dataset scripts are no longer supported" not in str(exc):
            raise
        parquet_path = ensure_fiqa_parquet()
        log(f"[dataset] using parquet fallback {parquet_path}")
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
    log(f"[dataset] loaded rows={len(rows)}")
    return rows


def add_precomputed_embeddings(rows: list[dict[str, Any]], model_name: str, batch_size: int) -> float:
    log(
        f"[embeddings] generating local embeddings model={model_name} rows={len(rows)} batch_size={batch_size}"
    )
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
    log(f"[embeddings] generated {len(rows)} embeddings")
    return elapsed


class ManticoreClient:
    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.pymysql = require_pymysql()
        self.host = host
        self.port = port
        self.timeout = timeout
        self.conn = None

    def connect(self):
        return self._connect_with_timeout(self.timeout)

    def _connect_with_timeout(self, timeout: float):
        if self.conn is None or not self.conn.open:
            self.conn = self.pymysql.connect(
                host=self.host,
                port=self.port,
                user="",
                password="",
                autocommit=True,
                charset="utf8mb4",
                connect_timeout=int(timeout),
                read_timeout=int(timeout),
                write_timeout=int(timeout),
            )
        return self.conn

    def wait_ready(self) -> None:
        log(f"[manticore] waiting for SQL on {self.host}:{self.port}")
        deadline = time.time() + self.timeout
        last_error = None
        while time.time() < deadline:
            try:
                conn = self.connect()
                with conn.cursor() as cursor:
                    cursor.execute("SHOW TABLES")
                    cursor.fetchall()
                log("[manticore] ready")
                return
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
                self.conn = None
            time.sleep(1)
        raise RuntimeError(f"Manticore SQL is not ready: {last_error}")

    def sql(self, statement: str, timeout_override: float | None = None) -> Any:
        if timeout_override is not None:
            if self.conn is not None and self.conn.open:
                self.conn.close()
            self.conn = self._connect_with_timeout(timeout_override)
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute(statement)
            return cursor.fetchall()

    def recreate_table(self, table_name: str, mode: str, num_dim: int | None) -> float:
        log(f"[manticore] recreating table={table_name} mode={mode}")
        started = time.perf_counter()
        ddl_timeout = schema_timeout_for_mode(mode, self.timeout)
        self.sql(f"DROP TABLE IF EXISTS {table_name}", timeout_override=ddl_timeout)
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
        self.sql(" ".join(ddl.split()), timeout_override=ddl_timeout)
        elapsed = time.perf_counter() - started
        log(f"[manticore] table={table_name} ready")
        return elapsed

    def import_rows(self, table_name: str, rows: list[dict[str, Any]], batch_size: int, mode: str) -> EngineResult:
        conn = self.connect()
        started = time.perf_counter()
        total_batches = max(1, math.ceil(len(rows) / batch_size))
        log(
            f"[manticore] import started table={table_name} rows={len(rows)} batch_size={batch_size} batches={total_batches}"
        )
        for batch_idx, batch in enumerate(chunked(rows, batch_size), start=1):
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
            inserted = min(batch_idx * batch_size, len(rows))
            progress = inserted / len(rows) * 100 if rows else 100.0
            if should_log_batch_progress(batch_idx, total_batches):
                log(
                    f"[manticore] imported {inserted}/{len(rows)} rows "
                    f"(batch {batch_idx}/{total_batches}, {progress:.1f}%)"
                )

        elapsed = time.perf_counter() - started
        log(f"[manticore] import finished in {elapsed:.4f}s")
        return EngineResult(
            seconds=elapsed,
            schema_setup_seconds=0.0,
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
        log(f"[typesense] waiting for HTTP on {self.base_url}")
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
                    log("[typesense] ready")
                    return
                last_error = response.text
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
            time.sleep(1)
        raise RuntimeError(f"Typesense is not ready: {last_error}")

    def recreate_collection(self, collection_name: str, mode: str, num_dim: int | None) -> float:
        log(f"[typesense] recreating collection={collection_name} mode={mode}")
        started = time.perf_counter()
        ddl_timeout = schema_timeout_for_mode(mode, self.timeout)
        self.requests.delete(
            f"{self.base_url}/collections/{collection_name}",
            headers=self.headers,
            timeout=ddl_timeout,
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
            timeout=max(ddl_timeout, 120),
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - started
        log(f"[typesense] collection={collection_name} ready")
        return elapsed

    def import_rows(self, collection_name: str, rows: list[dict[str, Any]], batch_size: int, mode: str) -> EngineResult:
        started = time.perf_counter()
        total_batches = max(1, math.ceil(len(rows) / batch_size))
        log(
            f"[typesense] import started collection={collection_name} rows={len(rows)} batch_size={batch_size} batches={total_batches}"
        )
        for batch_idx, batch in enumerate(chunked(rows, batch_size), start=1):
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
            inserted = min(batch_idx * batch_size, len(rows))
            progress = inserted / len(rows) * 100 if rows else 100.0
            if should_log_batch_progress(batch_idx, total_batches):
                log(
                    f"[typesense] imported {inserted}/{len(rows)} rows "
                    f"(batch {batch_idx}/{total_batches}, {progress:.1f}%)"
                )

        elapsed = time.perf_counter() - started
        log(f"[typesense] import finished in {elapsed:.4f}s")
        return EngineResult(
            seconds=elapsed,
            schema_setup_seconds=0.0,
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
        log(f"[dataset] reusing cached parquet {parquet_path}")
        return parquet_path

    log(f"[dataset] downloading parquet fallback to {parquet_path}")
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
    log(
        "[run] benchmark started "
        f"mode={args.mode} engines={args.engines} limit={args.limit} "
        f"batch_size={args.batch_size} embed_batch_size={args.embed_batch_size}"
    )
    if args.mode == "auto":
        log(
            "[run] auto mode enabled; each engine generates embeddings internally "
            f"(manticore={MANTICORE_AUTO_MODEL}, typesense={TYPESENSE_AUTO_MODEL})"
        )

    with StageTimer("dataset load"):
        rows = load_fiqa(args.dataset, args.limit)
    if not rows:
        raise RuntimeError("Dataset is empty")

    embedding_generation_seconds = 0.0
    if args.mode == "precomputed":
        with StageTimer("embedding generation"):
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
        with StageTimer("manticore import"):
            client.wait_ready()
            resource_monitor = DockerStatsMonitor(args.manticore_container, args.stats_interval)
            resource_monitor.start()
            try:
                schema_setup_seconds = client.recreate_table(args.collection, mode=args.mode, num_dim=num_dim)
                result = client.import_rows(args.collection, rows, args.batch_size, mode=args.mode)
            finally:
                resource_usage = resource_monitor.stop()
            result.schema_setup_seconds = schema_setup_seconds
            result.resource_usage = resource_usage
            log(f"[manticore] {format_resource_usage(resource_usage)}")
        output["manticore"] = asdict(result)

    if args.engines in ("typesense", "both"):
        client = TypesenseClient(args.typesense_url, args.typesense_api_key, timeout=args.timeout)
        with StageTimer("typesense import"):
            client.wait_ready()
            resource_monitor = DockerStatsMonitor(args.typesense_container, args.stats_interval)
            resource_monitor.start()
            try:
                schema_setup_seconds = client.recreate_collection(args.collection, mode=args.mode, num_dim=num_dim)
                result = client.import_rows(args.collection, rows, args.batch_size, mode=args.mode)
            finally:
                resource_usage = resource_monitor.stop()
            result.schema_setup_seconds = schema_setup_seconds
            result.resource_usage = resource_usage
            log(f"[typesense] {format_resource_usage(resource_usage)}")
        output["typesense"] = asdict(result)

    result_path = write_results(output, args.mode, len(rows))
    log(f"[run] benchmark finished result={result_path}")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {result_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
