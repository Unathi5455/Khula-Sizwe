"""Compute per-region, per-calendar-month baseline statistics."""

from __future__ import annotations

import pandas as pd


def compute_climatology(
    df: pd.DataFrame,
    value_col: str,
    baseline_start: str,
    baseline_end: str,
    group_col: str = "region_code",
    date_col: str = "month",
) -> pd.DataFrame:
    baseline = df[
        (df[date_col] >= baseline_start) & (df[date_col] <= baseline_end)
    ].copy()
    baseline["cal_month"] = pd.to_datetime(baseline[date_col]).dt.month

    stats = (
        baseline.groupby([group_col, "cal_month"])[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{value_col}_mean",
                "std": f"{value_col}_std",
                "count": f"{value_col}_n_obs",
            }
        )
    )
    return stats


def apply_anomaly(
    df: pd.DataFrame,
    climatology: pd.DataFrame,
    value_col: str,
    group_col: str = "region_code",
    date_col: str = "month",
    method: str = "zscore",
) -> pd.DataFrame:
    working = df.copy()
    working["cal_month"] = pd.to_datetime(working[date_col]).dt.month

    merged = working.merge(climatology, on=[group_col, "cal_month"], how="left")

    mean_col = f"{value_col}_mean"
    std_col = f"{value_col}_std"
    anomaly_col = f"{value_col}_anomaly"

    if method == "zscore":
        safe_std = merged[std_col].where(merged[std_col] > 0)
        merged[anomaly_col] = (merged[value_col] - merged[mean_col]) / safe_std
    elif method == "pct":
        safe_mean = merged[mean_col].where(merged[mean_col].abs() > 1e-9)
        merged[anomaly_col] = (merged[value_col] - merged[mean_col]) / safe_mean * 100
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'zscore' or 'pct'.")

    return merged.drop(columns=["cal_month"])
