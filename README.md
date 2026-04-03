# Upload Compare: Manticore vs Typesense

Мини-стенд для сравнения скорости загрузки одного и того же датасета в `Manticore Search` и `Typesense`, где эмбеддинги строятся самими движками при записи.

По умолчанию используется корпус `BeIR/fiqa` с Hugging Face (`57,638` документов). Для коротких прогонов можно ограничить размер через `--limit`.

## Что именно сравнивается

- `auto`: каждая система сама генерирует эмбеддинги при индексации.
  Manticore использует `sentence-transformers/all-MiniLM-L12-v2`, а Typesense использует встроенную модель `ts/all-MiniLM-L12-v2`.

На первом запуске в `auto` режиме обе системы будут скачивать модели. Такой прогон не стоит сравнивать с "прогретым" состоянием.

## Структура

- `docker-compose.yml` поднимает `Manticore` и `Typesense`
  Для `Manticore` используется образ `ghcr.io/manticoresoftware/manticoresearch:test-kit-fix_performance-mcl`, в котором `searchd` запускается явно через `entrypoint`/`command`, а не автоматически как PID 1.
- `scripts/benchmark.py`:
  - скачивает `FiQA`
  - при необходимости считает эмбеддинги
  - создает таблицу/коллекцию
  - загружает документы батчами
  - параллельно сэмплирует `docker stats` для CPU/RAM контейнера движка
  - пишет итог в `results/*.json`

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
python scripts/benchmark.py --mode auto --engines both --limit 1000
```

Host ports by default:

- Manticore SQL/HTTP: `19306` / `19308`
- Typesense HTTP: `18108`

Для benchmark Manticore теперь используется HTTP API на `19308`: схема создается через HTTP SQL endpoint, а документы загружаются батчами через `/bulk` в формате NDJSON.

Container names by default:

- Manticore: `upload-compare-manticore`
- Typesense: `upload-compare-typesense`

Особенность текущего Manticore-образа: сервис держится на процессе
`searchd --config /etc/manticoresearch/manticore.conf --nodetach`, который
задан в `docker-compose.yml` через `entrypoint`/`command`. Также штатный конфиг
в образе слушает `127.0.0.1`, поэтому в стенде он переопределён локальным
файлом [`manticore/manticore.conf`](/Users/djklim87/Projects/work/upload-ts-ms/manticore/manticore.conf)
с bind на `0.0.0.0`. Если убрать этот явный запуск или override-конфиг,
контейнер стартует, но `searchd` либо не поднимется автоматически, либо будет
доступен только внутри контейнера.

## GitHub Actions

В репозитории есть workflow `Upload Benchmark`, который:

- поднимает `Manticore` и `Typesense` через `docker compose`
- запускает benchmark
- пишет сводку прямо в `GitHub Actions Step Summary`
- прикладывает сырой `results/*.json` как artifact

По умолчанию CI гоняет `500` документов из `FiQA` в `auto` режиме, где эмбеддинги строятся самими движками. Полный корпус `57,638` документов оставлен для ручного `workflow_dispatch`, если нужен долгий прогон.

Параметры для ручного запуска через `workflow_dispatch`:

- `limit`
- `batch_size`
- `mode`

## Примеры

Сравнение нативной авто-генерации эмбеддингов:

```bash
python scripts/benchmark.py \
  --mode auto \
  --engines both \
  --limit 5000 \
  --batch-size 100
```

Только Manticore:

```bash
python scripts/benchmark.py --mode auto --engines manticore --limit 10000
```

## Что выводит benchmark

Примерно такой JSON:

```json
{
  "dataset": "BeIR/fiqa",
  "rows": 1000,
  "mode": "auto",
  "embedding_model": null,
  "embedding_generation_seconds": 0.0,
  "manticore": {
    "seconds": 1.73,
    "docs_per_second": 578.03,
    "resource_usage": {
      "cpu_percent_peak": 87.4,
      "memory_used_bytes_peak": 16252928
    }
  },
  "typesense": {
    "seconds": 2.11,
    "docs_per_second": 473.93,
    "resource_usage": {
      "cpu_percent_peak": 94.2,
      "memory_used_bytes_peak": 22478848
    }
  }
}
```

`resource_usage` содержит усредненные и пиковые значения по CPU и памяти, собранные через `docker stats` во время стадии schema setup + import. Если `docker` недоступен или контейнер назван иначе, benchmark не падает, а пишет `resource_usage.error`.

## Замечания по корректности

- Для Manticore в этом стенде загрузка идет через HTTP `/bulk`, а не через MySQL-протокол.
- Для честного сравнения прогоняй оба движка:
  - на одном и том же `--limit`
  - с одинаковым `--batch-size`
  - отдельно для cold и warm прогона

## Источники схем

- Manticore KNN / auto embeddings: https://manual.manticoresearch.com/Searching/KNN
- Typesense vector search / auto embedding: https://typesense.org/docs/30.1/api/vector-search.html
