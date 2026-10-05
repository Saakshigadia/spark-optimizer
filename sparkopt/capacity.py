"""Capacity planning: forecast data growth and storage needs.

Given a history of daily ingestion (GB per day), we fit two growth models:
  - linear:      ingestion grows by a fixed number of GB each day
  - exponential: ingestion grows by a fixed percentage each day
We keep whichever predicts a held-out slice of the history better, then simulate
the future day by day. Stored data = last `retention_days` of ingestion,
divided by the compression ratio and multiplied by the replication factor.
"""
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


@dataclass
class StoragePolicy:
    retention_days: int = 365
    compression_ratio: float = 3.0   # e.g. Parquet + Snappy is often 2-4x smaller than raw
    replication: int = 3             # HDFS default; use 1 for S3/GCS/ADLS
    capacity_tb: float = 100.0
    headroom: float = 0.2            # warn when usage passes 80% of capacity


def _mape(actual, predicted):
    actual = np.asarray(actual, float)
    return float(np.mean(np.abs((actual - predicted) / np.maximum(actual, 1e-9))))


def fit_growth(daily_gb):
    """Pick the better of a linear and an exponential trend. Returns (name, predict_fn, holdout_mape)."""
    y = np.asarray(daily_gb, float)
    if len(y) < 10:
        raise ValueError("Need at least 10 days of history")
    x = np.arange(len(y))
    split = int(len(y) * 0.8)

    def linear(xs, ys):
        a, b = np.polyfit(xs, ys, 1)
        return lambda t: np.maximum(a * np.asarray(t) + b, 0)

    def exponential(xs, ys):
        a, b = np.polyfit(xs, np.log(np.maximum(ys, 1e-6)), 1)
        return lambda t: np.exp(a * np.asarray(t) + b)

    scores = {}
    for name, fitter in (("linear", linear), ("exponential", exponential)):
        scores[name] = _mape(y[split:], fitter(x[:split], y[:split])(x[split:]))
    best = min(scores, key=scores.get)
    fitter = linear if best == "linear" else exponential
    return best, fitter(x, y), scores[best]


def forecast(daily_gb, start_date: date, months: int = 12, policy: StoragePolicy = StoragePolicy()):
    """Forecast monthly stored size and the date capacity runs out."""
    model, predict, mape = fit_growth(daily_gb)
    history = list(map(float, daily_gb))
    n = len(history)
    future_days = months * 30
    future = list(predict(np.arange(n, n + future_days)))
    series = history + future
    factor = policy.replication / policy.compression_ratio
    limit_tb = policy.capacity_tb * (1 - policy.headroom)

    monthly, warn_date, full_date = [], None, None
    for i in range(n, n + future_days):
        window = series[max(0, i - policy.retention_days + 1): i + 1]
        stored_tb = sum(window) * factor / 1024
        day = start_date + timedelta(days=i)
        if warn_date is None and stored_tb >= limit_tb:
            warn_date = day
        if full_date is None and stored_tb >= policy.capacity_tb:
            full_date = day
        if (i - n + 1) % 30 == 0:
            monthly.append({"month": day.strftime("%Y-%m"), "daily_ingest_gb": round(series[i], 1), "stored_tb": round(stored_tb, 2)})

    return {
        "model": model,
        "holdout_error_pct": round(mape * 100, 1),
        "current_daily_gb": round(history[-1], 1),
        "monthly": monthly,
        "warning_date": warn_date.isoformat() if warn_date else None,
        "full_date": full_date.isoformat() if full_date else None,
        "recommendation": _advice(monthly, warn_date, full_date, policy),
    }


def _advice(monthly, warn_date, full_date, policy):
    end = monthly[-1]["stored_tb"] if monthly else 0
    if full_date:
        extra = max(end - policy.capacity_tb, 0) * 1.25
        return (f"Storage fills up on {full_date.isoformat()}. Add about {extra:.0f} TB before then, "
                f"shorten retention below {policy.retention_days} days, or move cold data to cheaper storage.")
    if warn_date:
        return f"Usage passes {int((1 - policy.headroom) * 100)}% on {warn_date.isoformat()}. Plan an expansion before then."
    return f"Capacity is enough for the forecast period (about {end:.1f} TB of {policy.capacity_tb:.0f} TB used at the end)."
