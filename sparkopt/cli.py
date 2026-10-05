"""Command line interface.

  python -m sparkopt.cli analyze examples/bad_job.py
  python -m sparkopt.cli optimize --nodes 6 --cores 16 --memory 64 --input-gb 500 --job join --sla 30
  python -m sparkopt.cli capacity examples/ingestion_history.csv --months 12 --capacity-tb 160
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

from .ai import explain
from .analyzer import analyze, score
from .capacity import StoragePolicy, forecast
from .optimizer import Cluster, Workload, recommend


def read_history(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return date.fromisoformat(rows[0]["date"]), [float(r["ingested_gb"]) for r in rows]


def main(argv=None):
    p = argparse.ArgumentParser(prog="sparkopt", description="Spark optimization and capacity planning")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="find performance problems in a PySpark or SQL file")
    a.add_argument("file")

    o = sub.add_parser("optimize", help="recommend Spark resources for a workload")
    o.add_argument("--nodes", type=int, required=True)
    o.add_argument("--cores", type=int, required=True, help="cores per node")
    o.add_argument("--memory", type=int, required=True, help="GB of memory per node")
    o.add_argument("--input-gb", type=float, required=True)
    o.add_argument("--job", default="etl", choices=["etl", "aggregation", "join", "ml"])
    o.add_argument("--sla", type=float, default=60, help="SLA in minutes")

    c = sub.add_parser("capacity", help="forecast storage from a CSV of date,ingested_gb")
    c.add_argument("csv")
    c.add_argument("--months", type=int, default=12)
    c.add_argument("--retention-days", type=int, default=365)
    c.add_argument("--compression", type=float, default=3.0)
    c.add_argument("--replication", type=int, default=3)
    c.add_argument("--capacity-tb", type=float, default=100)

    args = p.parse_args(argv)
    if args.cmd == "analyze":
        code = Path(args.file).read_text()
        lang = "sql" if args.file.endswith(".sql") else "python"
        findings = analyze(code, lang)
        print(f"Health score: {score(findings)}/100  ({len(findings)} issue(s))\n")
        for f in findings:
            print(f"[{f.severity.upper():6}] line {f.line}: {f.title}\n         why: {f.why}\n         fix: {f.fix}\n")
        print(explain(findings, code)["text"])
    elif args.cmd == "optimize":
        r = recommend(Cluster(args.nodes, args.cores, args.memory), Workload(args.input_gb, args.job, args.sla))
        print(json.dumps({k: r[k] for k in ("executors", "conf", "estimated_minutes", "sla_met", "nodes_needed")}, indent=2))
        print("\n" + "\n".join("- " + n for n in r["notes"]))
        print("\n" + r["spark_submit"])
    else:
        start, hist = read_history(args.csv)
        pol = StoragePolicy(args.retention_days, args.compression, args.replication, args.capacity_tb)
        print(json.dumps(forecast(hist, start, args.months, pol), indent=2))


if __name__ == "__main__":
    main()
