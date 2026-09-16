"""Load the manually-exported MODIS NDVI series and compute VCI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.utils import get_logger, load_regions, load_sources

LOG = get_logger("processing.ndvi")


def load_raw_ndvi(path: Path | None = None) -> pd.DataFrame:
    config = load_sources()
    cfg = config["ndvi"]

    if path is None:
        path = Path(config["paths"]["raw_ndvi"]) / cfg["raw_filename"]

    if not path.exists():
        raise FileNotFoundError(
            f"No NDVI export found at {path}. Pull it from NASA AppEEARS first "
            "(see README) and drop the CSV there."
        )

    raw = pd.read_csv(path)
    missing_cols = {cfg["date_column"], cfg["region_column"], cfg["value_column"]} - set(
        raw.columns
    )
    if missing_cols:
        raise ValueError(
            f"NDVI export is missing expected column(s) {missing_cols}. "
            f"Available columns: {list(raw.columns)}."
        )

    tidy = raw[[cfg["region_column"], cfg["date_column"], cfg["value_column"]]].copy()
    tidy.columns = ["region_code", "date", "ndvi_raw"]
    tidy["date"] = pd.to_datetime(tidy["date"])

    tidy.loc[tidy["ndvi_raw"] <= cfg["fill_value"], "ndvi_raw"] = pd.NA
    tidy["ndvi"] = tidy["ndvi_raw"].astype("Float64") * cfg["scale_factor"]

    valid_codes = {r["code"] for r in load_regions()}
    unknown = set(tidy["region_code"].unique()) - valid_codes
    if unknown:
        raise ValueError(
            f"NDVI export has region codes not in config/regions.yml: {unknown}."
        )

    return tidy.drop(columns=["ndvi_raw"])


def aggregate_to_monthly(ndvi_16day: pd.DataFrame) -> pd.DataFrame:
    working = ndvi_16day.copy()
    working["month"] = working["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        working.groupby(["region_code", "month"], as_index=False)
        .agg(ndvi=("ndvi", "mean"), composites_observed=("ndvi", "count"))
        .sort_values(["region_code", "month"])
        .reset_index(drop=True)
    )
    return monthly


def compute_vci(
    monthly: pd.DataFrame,
    baseline_start: str,
    baseline_end: str,
    group_col: str = "region_code",
    date_col: str = "month",
) -> pd.DataFrame:
    working = monthly.copy()
    working["cal_month"] = pd.to_datetime(working[date_col]).dt.month

    baseline = working[
        (working[date_col] >= baseline_start) & (working[date_col] <= baseline_end)
    ]
    ranges = (
        baseline.groupby([group_col, "cal_month"])["ndvi"]
        .agg(ndvi_min="min", ndvi_max="max", n_obs="count")
        .reset_index()
    )

    merged = working.merge(ranges, on=[group_col, "cal_month"], how="left")
    span = (merged["ndvi_max"] - merged["ndvi_min"]).where(
        lambda s: s.abs() > 1e-9
    )
    merged["vci"] = (merged["ndvi"] - merged["ndvi_min"]) / span * 100
    merged["vci"] = merged["vci"].clip(lower=0, upper=100)

    return merged.drop(columns=["cal_month"])


def build_vci_table(
    baseline_start: str, baseline_end: str, path: Path | None = None
) -> pd.DataFrame:
    raw = load_raw_ndvi(path)
    monthly = aggregate_to_monthly(raw)
    return compute_vci(monthly, baseline_start, baseline_end)
