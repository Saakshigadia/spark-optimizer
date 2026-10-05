from datetime import date
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from sparkopt import ai
from sparkopt.analyzer import analyze, analyze_python, analyze_sql, score
from sparkopt.capacity import StoragePolicy, fit_growth, forecast
from sparkopt.optimizer import Cluster, Workload, recommend, size_executors

ROOT = Path(__file__).resolve().parents[1]
rules = lambda fs: {f.rule for f in fs}


# ---------- analyser
def test_bad_job_has_the_expected_problems():
    found = rules(analyze_python((ROOT / "examples/bad_job.py").read_text()))
    assert {"python-udf", "collect", "global-sort", "no-cache", "aqe-off", "broadcast-off", "default-partitions"} <= found


def test_optimized_job_is_clean_of_high_severity_issues():
    fs = analyze_python((ROOT / "examples/optimized_job.py").read_text())
    assert not [f for f in fs if f.severity == "high"]


def test_decorator_udf_reported_once():
    code = "from pyspark.sql.functions import udf\n@udf('double')\ndef f(x):\n    return x\n"
    assert [f.rule for f in analyze_python(code)].count("python-udf") == 1


def test_inline_udf_and_loops():
    code = "f = udf(lambda x: x + 1)\nfor c in cols:\n    df = df.withColumn(c, F.lit(1))\n    df.count()\n"
    assert {"python-udf", "with-column-loop", "action-in-loop"} <= rules(analyze_python(code))


def test_comments_and_strings_do_not_trigger():
    assert analyze_python("# df.collect()\nmsg = 'never call toPandas()'\n") == []


def test_cached_dataframe_not_flagged():
    code = "df = spark.read.parquet('x').cache()\ndf.count()\ndf.show()\n"
    assert "no-cache" not in rules(analyze_python(code))


def test_sql_rules_and_line_numbers():
    fs = analyze_sql((ROOT / "examples/report_query.sql").read_text())
    by_rule = {f.rule: f.line for f in fs}
    assert {"select-star", "implicit-join", "not-in-subquery", "function-on-filter", "union-distinct", "order-no-limit"} <= set(by_rule)
    assert by_rule["order-no-limit"] == 9


def test_sql_clean_query():
    sql = "SELECT id, amount FROM orders WHERE dt >= '2026-01-01' ORDER BY amount DESC LIMIT 10"
    assert analyze_sql(sql) == []


def test_language_autodetect_and_score():
    assert analyze("SELECT * FROM t")[0].rule == "select-star"
    assert score([]) == 100 and score(analyze_python("df.collect()\n")) == 85


def test_syntax_error_is_reported_not_raised():
    assert analyze_python("def broken(:\n")[0].rule == "syntax"


# ---------- optimizer
def test_executor_sizing_matches_the_standard_rule():
    # classic example: 6 nodes x 16 cores x 64 GB -> 17 executors, 5 cores, 18 GB each
    ex = size_executors(Cluster(6, 16, 64))
    assert ex["num_executors"] == 17 and ex["executor_cores"] == 5 and ex["executor_memory_gb"] == 18


def test_partitions_fill_whole_waves():
    r = recommend(Cluster(6, 16, 64), Workload(500, "join", 60))
    assert r["conf"]["spark.sql.shuffle.partitions"] % r["executors"]["total_cores"] == 0
    assert "spark.sql.autoBroadcastJoinThreshold" in r["conf"]


def test_sla_miss_suggests_more_nodes():
    r = recommend(Cluster(2, 8, 32), Workload(2000, "join", 10))
    assert not r["sla_met"] and r["nodes_needed"] > 2


def test_small_nodes_do_not_break():
    ex = size_executors(Cluster(1, 2, 4))
    assert ex["num_executors"] >= 1 and ex["executor_memory_gb"] >= 1


def test_unknown_job_type():
    with pytest.raises(ValueError):
        recommend(Cluster(2, 8, 32), Workload(10, "streaming"))


# ---------- capacity
def test_picks_linear_for_linear_growth():
    model, predict, err = fit_growth([100 + 2 * i for i in range(60)])
    assert model == "linear" and err < 0.01 and abs(predict(70) - 240) < 1


def test_picks_exponential_for_compound_growth():
    model, _, _ = fit_growth([100 * 1.02 ** i for i in range(90)])
    assert model == "exponential"


def test_forecast_finds_the_day_storage_fills():
    hist = [100 + i for i in range(100)]
    out = forecast(hist, date(2026, 1, 1), 12, StoragePolicy(retention_days=365, compression_ratio=1, replication=1, capacity_tb=40))
    assert out["full_date"] is not None and len(out["monthly"]) == 12
    assert out["monthly"][-1]["stored_tb"] > out["monthly"][0]["stored_tb"]


def test_forecast_needs_history():
    with pytest.raises(ValueError):
        forecast([1, 2, 3], date(2026, 1, 1))


# ---------- AI layer
def test_ai_falls_back_to_rules_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = ai.explain(analyze_python("df.collect()\n"), "df.collect()")
    assert out["source"] == "rules" and "collect" in out["text"]


def test_ai_uses_llm_response_when_key_set(monkeypatch):
    import io, json
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    fake = io.BytesIO(json.dumps({"content": [{"type": "text", "text": "Use broadcast."}]}).encode())
    fake.__enter__ = lambda s=fake: s
    fake.__exit__ = lambda *a: None
    monkeypatch.setattr(ai.urllib.request, "urlopen", lambda req, timeout: fake)
    out = ai.explain([], "x")
    assert out == {"source": "llm", "model": ai.DEFAULT_MODEL, "text": "Use broadcast."}


def test_ai_survives_network_errors(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    def boom(*a, **k): raise OSError("offline")
    monkeypatch.setattr(ai.urllib.request, "urlopen", boom)
    assert ai.explain([], "x")["source"] == "rules"


# ---------- API
def test_api_endpoints():
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/analyze", json={"code": "df.collect()\n"}).json()
    assert r["score"] == 85 and r["findings"][0]["rule"] == "collect"
    r = c.post("/optimize", json={"nodes": 6, "cores_per_node": 16, "memory_gb_per_node": 64, "input_gb": 500, "job_type": "join"})
    assert r.status_code == 200 and r.json()["executors"]["num_executors"] == 17
    hist = list(np.linspace(100, 200, 60))
    r = c.post("/capacity", json={"start_date": "2026-01-01", "daily_gb": hist, "months": 6})
    assert r.status_code == 200 and len(r.json()["monthly"]) == 6
    assert c.post("/capacity", json={"start_date": "2026-01-01", "daily_gb": [1, 2]}).status_code == 422
