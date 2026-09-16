"""Standardized Precipitation Index (McKee et al., 1993)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rolling_accumulation(
    df: pd.DataFrame,
    value_col: str,
    window: int,
    group_col: str = "region_code",
    date_col: str = "month",
) -> pd.Series:
    working = df.sort_values([group_col, date_col])
    _assert_no_gaps(working, group_col, date_col)

    accumulated = working.groupby(group_col)[value_col].transform(
        lambda s: s.rolling(window, min_periods=window).sum()
    )
    return accumulated.reindex(df.index)


def _assert_no_gaps(df: pd.DataFrame, group_col: str, date_col: str) -> None:
    for region, group in df.groupby(group_col):
        months = pd.PeriodIndex(pd.to_datetime(group[date_col]), freq="M")
        expected = pd.period_range(months.min(), months.max(), freq="M")
        missing = expected.difference(months)
        if len(missing):
            raise ValueError(
                f"Region {region} has {len(missing)} missing month(s) in its "
                f"series, e.g. {missing[0]}. Fill or drop gaps before "
                "computing SPI."
            )


def _fit_gamma_by_calendar_month(
    accumulated: pd.Series,
    cal_months: pd.Series,
    min_obs: int = 10,
) -> dict[int, dict[str, float]]:
    params: dict[int, dict[str, float]] = {}

    for cal_month in range(1, 13):
        values = accumulated[cal_months == cal_month].dropna()
        if len(values) < min_obs:
            continue

        zeros = (values == 0).sum()
        q = zeros / len(values)
        nonzero = values[values > 0]

        if len(nonzero) < max(5, min_obs // 2):
            continue

        shape, loc, scale = stats.gamma.fit(nonzero, floc=0)
        params[cal_month] = {"q": q, "shape": shape, "loc": loc, "scale": scale}

    return params


def _transform_to_spi(
    value: float, cal_month: int, params: dict[int, dict[str, float]]
) -> float:
    if pd.isna(value) or cal_month not in params:
        return np.nan

    p = params[cal_month]
    if value <= 0:
        cumulative_prob = p["q"]
    else:
        gamma_cdf = stats.gamma.cdf(value, p["shape"], loc=p["loc"], scale=p["scale"])
        cumulative_prob = p["q"] + (1 - p["q"]) * gamma_cdf

    cumulative_prob = np.clip(cumulative_prob, 1e-6, 1 - 1e-6)
    return stats.norm.ppf(cumulative_prob)


def compute_spi(
    df: pd.DataFrame,
    window: int,
    baseline_start: str,
    baseline_end: str,
    rainfall_col: str = "rainfall_mm",
    group_col: str = "region_code",
    date_col: str = "month",
) -> pd.DataFrame:
    working = df.sort_values([group_col, date_col]).reset_index(drop=True)
    working[f"spi_accum_{window}"] = rolling_accumulation(
        working, rainfall_col, window, group_col, date_col
    )
    working["_cal_month"] = pd.to_datetime(working[date_col]).dt.month

    spi_col = f"spi_{window}"
    working[spi_col] = np.nan

    for region, group in working.groupby(group_col):
        baseline_mask = (group[date_col] >= baseline_start) & (
            group[date_col] <= baseline_end
        )
        params = _fit_gamma_by_calendar_month(
            group.loc[baseline_mask, f"spi_accum_{window}"],
            group.loc[baseline_mask, "_cal_month"],
        )
        if not params:
            continue

        region_spi = group.apply(
            lambda row: _transform_to_spi(
                row[f"spi_accum_{window}"], row["_cal_month"], params
            ),
            axis=1,
        )
        working.loc[group.index, spi_col] = region_spi

    return working.drop(columns=["_cal_month"])


def classify_spi(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value >= 2.0:
        return "extremely_wet"
    if value >= 1.5:
        return "very_wet"
    if value >= 1.0:
        return "moderately_wet"
    if value > -1.0:
        return "near_normal"
    if value > -1.5:
        return "moderately_dry"
    if value > -2.0:
        return "severely_dry"
    return "extremely_dry"
