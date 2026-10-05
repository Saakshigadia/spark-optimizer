"""Build docs/index.html (the GitHub Pages demo) from docs/template.html.
Embeds the sample code, the sample ingestion history and the latest benchmark results.
Run from the project root:  python scripts/export_web.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with open(ROOT / "examples/ingestion_history.csv") as f:
        rows = list(csv.DictReader(f))
    data = {
        "samples": {
            "py": (ROOT / "examples/bad_job.py").read_text(),
            "good": (ROOT / "examples/optimized_job.py").read_text(),
            "sql": (ROOT / "examples/report_query.sql").read_text(),
        },
        "history": {"start": rows[0]["date"], "values": [float(r["ingested_gb"]) for r in rows]},
    }
    bench = ROOT / "outputs/benchmark.json"
    if bench.exists():
        data["benchmark"] = json.loads(bench.read_text())
        try:
            import pyspark
            data["spark_version"] = pyspark.__version__
        except ImportError:
            # version recorded when the benchmark was run
            if "spark_version" in data["benchmark"]:
                data["spark_version"] = data["benchmark"]["spark_version"]
    html = (ROOT / "docs/template.html").read_text().replace("/*__DATA__*/{}", json.dumps(data))
    (ROOT / "docs/index.html").write_text(html)
    print("Wrote docs/index.html")


if __name__ == "__main__":
    main()
