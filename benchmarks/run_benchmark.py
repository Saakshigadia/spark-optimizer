"""Time the original job against the optimised job on the same data.

Runs Spark in local mode, so absolute numbers depend on your machine;
the comparison between the two versions is what matters.
Usage:  python benchmarks/run_benchmark.py [rows]
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyspark.sql import SparkSession, functions as F  # noqa: E402
from examples import bad_job, optimized_job  # noqa: E402


def make_data(spark, rows, path):
    orders = (spark.range(rows).withColumnRenamed("id", "order_id")
              .withColumn("country_id", (F.col("order_id") * 7919 % 60).cast("int"))
              .withColumn("amount", F.round(F.rand(seed=1) * 5000 + 50, 2)))
    regions = ["Asia", "Europe", "Africa", "Americas", "Oceania"]
    countries = spark.createDataFrame([(i, regions[i % 5], [0.05, 0.12, 0.18, 0.2][i % 4]) for i in range(60)],
                                      "country_id int, region string, tax_rate double")
    orders.write.mode("overwrite").parquet(f"{path}/orders")
    countries.write.mode("overwrite").parquet(f"{path}/countries")


def main():
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    spark = (SparkSession.builder.master("local[*]").appName("sparkopt-benchmark")
             .config("spark.ui.enabled", "false").config("spark.driver.memory", "2g").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    path = "/tmp/sparkopt_bench"
    make_data(spark, rows, path)
    orders, countries = spark.read.parquet(f"{path}/orders"), spark.read.parquet(f"{path}/countries")

    timings = {"original": [], "optimized": []}
    results = {}
    bad_job.run(spark, orders, countries); optimized_job.run(spark, orders, countries)   # warm-up (JIT, caches)
    for _ in range(3):
        for name, job in (("original", bad_job), ("optimized", optimized_job)):
            t = time.perf_counter()
            results[name] = job.run(spark, orders, countries)
            timings[name].append(time.perf_counter() - t)
            print(f"{name}: {timings[name][-1]:.1f}s", flush=True)
    spark.stop()

    a, b = results["original"], results["optimized"]
    same = a[0] == b[0] and [x[0] for x in a[1]] == [x[0] for x in b[1]] and \
        all(abs(x[1] - y[1]) < 1 for x, y in zip(a[1], b[1])) and set(a[2]) == set(b[2])
    med = {k: round(statistics.median(v), 2) for k, v in timings.items()}
    out = {"rows": rows, "median_seconds": med, "runs": {k: [round(x, 2) for x in v] for k, v in timings.items()},
           "speedup": round(med["original"] / med["optimized"], 1),
           "time_saved_pct": round(100 * (1 - med["optimized"] / med["original"]), 1),
           "same_results": same}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
