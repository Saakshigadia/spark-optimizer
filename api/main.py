"""REST API.   uvicorn api.main:app --reload   ->   http://127.0.0.1:8000/docs"""
from datetime import date
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sparkopt.ai import explain
from sparkopt.analyzer import analyze, score
from sparkopt.capacity import StoragePolicy, forecast
from sparkopt.optimizer import Cluster, Workload, recommend

app = FastAPI(title="Spark Optimization & Capacity Planning API", version="1.0.0")


class AnalyzeRequest(BaseModel):
    code: str = Field(..., min_length=1)
    language: Optional[Literal["python", "sql"]] = None
    explain: bool = True


class OptimizeRequest(BaseModel):
    nodes: int = Field(..., ge=1)
    cores_per_node: int = Field(..., ge=1)
    memory_gb_per_node: int = Field(..., ge=2)
    input_gb: float = Field(..., gt=0)
    job_type: Literal["etl", "aggregation", "join", "ml"] = "etl"
    sla_minutes: float = Field(60, gt=0)


class CapacityRequest(BaseModel):
    start_date: date
    daily_gb: List[float] = Field(..., min_length=10)
    months: int = Field(12, ge=1, le=60)
    retention_days: int = Field(365, ge=1)
    compression_ratio: float = Field(3.0, gt=0)
    replication: int = Field(3, ge=1)
    capacity_tb: float = Field(100, gt=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze_code(req: AnalyzeRequest):
    findings = analyze(req.code, req.language)
    out = {"score": score(findings), "findings": [f.to_dict() for f in findings]}
    if req.explain:
        out["explanation"] = explain(findings, req.code)
    return out


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    return recommend(Cluster(req.nodes, req.cores_per_node, req.memory_gb_per_node),
                     Workload(req.input_gb, req.job_type, req.sla_minutes))


@app.post("/capacity")
def capacity(req: CapacityRequest):
    try:
        return forecast(req.daily_gb, req.start_date, req.months,
                        StoragePolicy(req.retention_days, req.compression_ratio, req.replication, req.capacity_tb))
    except ValueError as e:
        raise HTTPException(422, str(e))
