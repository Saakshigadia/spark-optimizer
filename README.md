# Spark Optimization & Capacity Planning Platform

A toolkit for data engineers that answers three questions:

1. **Why is my Spark job slow?** A code analyser finds common performance problems in PySpark
   scripts and Spark SQL, explains each one and suggests a fix. An optional AI layer (Claude)
   writes a plain-language explanation and rewritten code.
2. **What resources does it need?** A resource optimizer recommends executors, cores, memory and
   shuffle partitions for a cluster and workload, estimates runtime and checks it against an SLA.
3. **When will storage run out?** A capacity planner forecasts data growth from ingestion history
   and tells you the date storage fills up.

**Live demo:** https://saakshigadia.github.io/spark-optimizer/

## Result

The analyser was run on a typical first version of a sales report job (`examples/bad_job.py`).
Applying its suggestions gave `examples/optimized_job.py`. Both were timed on the same 500,000
orders with PySpark 4.2 in local mode (median of 3 runs, `outputs/benchmark.json`):

| | Original | Optimised |
|---|---|---|
| Runtime | 47.0 s | **1.1 s** |
| Health score | 43 / 100 | **100 / 100** |
| Results | | identical |

What changed: the Python UDF became a built-in expression, the small countries table is broadcast
instead of shuffled, the DataFrame used by three actions is cached, the top-10 query uses
`orderBy().limit()` instead of collecting every row to the driver, AQE is on, and shuffle
partitions are sized for the data. Most of the gain comes from removing the Python UDF and the
full `collect()`. On a real cluster the exact speed-up will differ, but these are the same
patterns that slow production jobs.

## Code analyser

PySpark code is parsed with Python's `ast` module, so rules match real calls, not comments or
strings. SQL is checked after comments and string literals are removed.

| PySpark rules | SQL rules |
|---|---|
| Python UDFs | `SELECT *` |
| `collect()` / `toPandas()` on full data | Comma joins (`FROM a, b`) and `CROSS JOIN` |
| `crossJoin`, `join` without a key | `NOT IN (subquery)` |
| RDD `groupByKey` | `UNION` instead of `UNION ALL` |
| `repartition(1)` / `coalesce(1)` | Functions on filtered columns (blocks partition pruning) |
| Global sort without `limit` | `ORDER BY` without `LIMIT` |
| `withColumn` or actions inside loops | Leading-wildcard `LIKE` |
| DataFrame reused by several actions without `cache()` | `COUNT(DISTINCT)` |
| AQE disabled, broadcast disabled, default 200 partitions | |
| Missing broadcast hint, `inferSchema=True` | |

Each finding has a severity, line number, explanation and fix, and the file gets a 0-100 health score.

**AI layer:** set `ANTHROPIC_API_KEY` and the findings plus code are sent to Claude, which
explains the biggest bottlenecks and rewrites those parts. Without a key, or if the API call
fails, a rule-based summary is used so the tool always works.

## Resource optimizer

Uses the standard Spark-on-YARN sizing rules: about 5 cores per executor, 1 core and 1 GB per node
reserved for the OS, memory overhead of max(384 MB, 10%), one executor slot for the driver, and
shuffle partitions of about 128 MB rounded to full waves of tasks. For example, 6 nodes × 16 cores
× 64 GB gives 17 executors with 5 cores and 18 GB each.

Runtime is estimated with a simple throughput model per job type (ETL, aggregation, join, ML).
If the estimate misses the SLA, it suggests how many nodes would meet it. The throughput numbers
are starting points: calibrate them with timings from your own jobs.

## Capacity planner

Fits a linear and an exponential growth model to daily ingestion, keeps the one that predicts a
held-out 20% of the history better, then simulates storage day by day using retention,
compression and replication. It reports monthly stored size, the date usage crosses the warning
threshold, the date storage is full, and how much to add.

## Run it

```bash
pip install -r requirements.txt

python -m sparkopt.cli analyze examples/bad_job.py
python -m sparkopt.cli analyze examples/report_query.sql
python -m sparkopt.cli optimize --nodes 6 --cores 16 --memory 64 --input-gb 500 --job join --sla 30
python -m sparkopt.cli capacity examples/ingestion_history.csv --capacity-tb 160

pytest -q                               # 23 tests
uvicorn api.main:app --reload           # API docs at http://127.0.0.1:8000/docs
python benchmarks/run_benchmark.py 500000   # needs PySpark and Java 17+
python scripts/export_web.py            # rebuild the demo in docs/
```

The demo is rebuilt and deployed to GitHub Pages automatically on every push (`.github/workflows/pages.yml`).

## API

| Method | Route | What it does |
|---|---|---|
| POST | `/analyze` | `{code, language?, explain?}` → findings, score, explanation |
| POST | `/optimize` | cluster + workload → Spark config, runtime estimate, SLA check |
| POST | `/capacity` | daily ingestion + policy → monthly forecast and dates |

## Project structure

```
sparkopt/analyzer.py    PySpark (AST) and SQL rules
sparkopt/optimizer.py   executor sizing, partitions, runtime estimate
sparkopt/capacity.py    growth model selection and storage forecast
sparkopt/ai.py          Claude explanations with rule-based fallback
sparkopt/cli.py         command line
api/main.py             FastAPI service
examples/               slow job, optimised job, SQL query, ingestion history
benchmarks/             timing script, results in outputs/benchmark.json
docs/                   browser demo (GitHub Pages)
tests/                  23 pytest tests
```

## Limitations and next steps

- The browser demo uses simplified pattern matching; the Python tool is more accurate.
- Read Spark event logs to find skewed stages and spill from real runs.
- Calibrate the runtime model automatically from past job history.
