# Upload Compare: Manticore vs Typesense

Мини-стенд для сравнения скорости загрузки одного и того же датасета в `Manticore Search` и `Typesense` с векторным полем.

По умолчанию используется корпус `BeIR/fiqa` с Hugging Face (`57,638` документов). Для коротких прогонов можно ограничить размер через `--limit`.

## Что именно сравнивается

- `auto`: каждая система сама генерирует эмбеддинги при индексации. Это теперь дефолтный сценарий для CI:
  Manticore использует `sentence-transformers/all-MiniLM-L12-v2`, а Typesense использует встроенную модель `ts/all-MiniLM-L12-v2`.

- `precomputed`: эмбеддинги строятся локально в Python одной и той же моделью `sentence-transformers/all-MiniLM-L6-v2`, после чего одинаковые векторы загружаются и в Manticore, и в Typesense. Этот режим оставлен как отдельный сценарий, если нужно сравнить именно ingestion без стоимости встроенного embedding pipeline.

На первом запуске в `auto` режиме обе системы будут скачивать модели. Такой прогон не стоит сравнивать с "прогретым" состоянием.

## Структура

- `docker-compose.yml` поднимает `Manticore` и `Typesense`
- `scripts/benchmark.py`:
  - скачивает `FiQA`
  - при необходимости считает эмбеддинги
  - создает таблицу/коллекцию
  - загружает документы батчами
  - пишет итог в `results/*.json`

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
python scripts/benchmark.py --mode precomputed --engines both --limit 1000
```

Host ports by default:

- Manticore SQL/HTTP: `19306` / `19308`
- Typesense HTTP: `18108`

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
- `embed_batch_size`
- `mode`

## Примеры

Честное сравнение ingestion:

```bash
python scripts/benchmark.py \
  --mode precomputed \
  --engines both \
  --limit 10000 \
  --embed-model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 250
```

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
python scripts/benchmark.py --mode precomputed --engines manticore --limit 10000
```

## Что выводит benchmark

Примерно такой JSON:

```json
{
  "dataset": "BeIR/fiqa",
  "rows": 1000,
  "mode": "precomputed",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_generation_seconds": 2.84,
  "manticore": {
    "seconds": 1.73,
    "docs_per_second": 578.03
  },
  "typesense": {
    "seconds": 2.11,
    "docs_per_second": 473.93
  }
}
```

## Замечания по корректности

- Если цель именно сравнить скорость загрузки, используй `precomputed`.
- Если цель сравнить "как быстро система поднимает semantic-ready индекс", используй `auto`.
- Для честного сравнения прогоняй оба движка:
  - на одном и том же `--limit`
  - с одинаковым `--batch-size`
  - отдельно для cold и warm прогона

## Источники схем

- Manticore KNN / auto embeddings: https://manual.manticoresearch.com/Searching/KNN
- Typesense vector search / auto embedding: https://typesense.org/docs/30.1/api/vector-search.html
