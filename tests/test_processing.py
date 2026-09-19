"""Tests for the processing layer, using synthetic data so no network is needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.climatology import apply_anomaly, compute_climatology
from src.processing.ndvi import aggregate_to_monthly, compute_vci
from src.processing.spi import (
    _assert_no_gaps,
    classify_spi,
    compute_spi,
    rolling_accumulation,
)


def _synthetic_rainfall(n_years: int = 15, seed: int = 0) -> pd.DataFrame:
    """Region with a clear wet-summer/dry-winter seasonal cycle, gamma-ish rain."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2005-01-01", periods=n_years * 12, freq="MS")
    seasonal = np.cos(2 * np.pi * (months.month - 1) / 12)  # peak in Jan
    means = np.clip(60 + 50 * seasonal, 5, None)
    rainfall = rng.gamma(shape=2.0, scale=means / 2.0)

    return pd.DataFrame(
        {"region_code": "GT", "region_name": "Gauteng", "month": months,
         "rainfall_mm": rainfall}
    )


def test_climatology_mean_reflects_seasonal_cycle():
    df = _synthetic_rainfall()
    clim = compute_climatology(df, "rainfall_mm", "2005-01-01", "2019-12-31")

    jan_mean = clim.loc[clim["cal_month"] == 1, "rainfall_mm_mean"].iloc[0]
    jun_mean = clim.loc[clim["cal_month"] == 6, "rainfall_mm_mean"].iloc[0]
    assert jan_mean > jun_mean


def test_apply_anomaly_zscore_centers_baseline_near_zero():
    df = _synthetic_rainfall()
    clim = compute_climatology(df, "rainfall_mm", "2005-01-01", "2019-12-31")
    result = apply_anomaly(df, clim, "rainfall_mm", method="zscore")

    assert "rainfall_mm_anomaly" in result.columns
    assert abs(result["rainfall_mm_anomaly"].mean()) < 0.5


def test_apply_anomaly_handles_zero_std_without_error():
    df = pd.DataFrame(
        {
            "region_code": ["A"] * 24,
            "month": pd.date_range("2020-01-01", periods=24, freq="MS"),
            "value": [5.0] * 24,
        }
    )
    clim = compute_climatology(df, "value", "2020-01-01", "2021-12-31")
    result = apply_anomaly(df, clim, "value", method="zscore")
    assert result["value_anomaly"].isna().all()


def test_apply_anomaly_rejects_unknown_method():
    df = _synthetic_rainfall()
    clim = compute_climatology(df, "rainfall_mm", "2005-01-01", "2019-12-31")
    with pytest.raises(ValueError):
        apply_anomaly(df, clim, "rainfall_mm", method="bogus")


def test_rolling_accumulation_matches_manual_sum():
    df = _synthetic_rainfall(n_years=3)
    result = rolling_accumulation(df, "rainfall_mm", window=3)
    manual = df["rainfall_mm"].rolling(3, min_periods=3).sum()
    pd.testing.assert_series_equal(result, manual, check_names=False)


def test_rolling_accumulation_detects_gap():
    df = _synthetic_rainfall(n_years=2)
    with_gap = df.drop(index=5).reset_index(drop=True)
    with pytest.raises(ValueError, match="missing"):
        rolling_accumulation(with_gap, "rainfall_mm", window=3)


def test_spi_is_roughly_standard_normal_on_baseline():
    df = _synthetic_rainfall(n_years=20)
    result = compute_spi(df, window=3, baseline_start="2005-01-01",
                          baseline_end="2024-12-31")

    spi = result["spi_3"].dropna()
    assert len(spi) > 100
    assert abs(spi.mean()) < 0.3
    assert 0.7 < spi.std() < 1.3
    assert spi.between(-4, 4).all()


def test_spi_flags_a_known_dry_spell():
    df = _synthetic_rainfall(n_years=15)
    dry_start = df.index[df["month"] == "2015-01-01"][0]
    df.loc[dry_start:dry_start + 2, "rainfall_mm"] = 0.1

    result = compute_spi(df, window=3, baseline_start="2005-01-01",
                          baseline_end="2019-12-31")
    dry_spi = result.loc[dry_start + 2, "spi_3"]
    assert dry_spi < -1.0


def test_spi_returns_nan_when_baseline_too_thin():
    df = _synthetic_rainfall(n_years=15)
    result = compute_spi(
        df, window=3, baseline_start="2005-01-01", baseline_end="2005-06-30"
    )
    assert result["spi_3"].isna().all()


def test_classify_spi_buckets():
    assert classify_spi(np.nan) == "unknown"
    assert classify_spi(2.5) == "extremely_wet"
    assert classify_spi(0.0) == "near_normal"
    assert classify_spi(-1.2) == "moderately_dry"
    assert classify_spi(-2.5) == "extremely_dry"


def test_vci_bounds_and_known_extremes():
    rng = np.random.default_rng(1)
    months = pd.date_range("2010-01-01", periods=60, freq="MS")
    seasonal = 0.5 + 0.1 * np.sin(2 * np.pi * months.month / 12)
    noise = rng.normal(0, 0.03, len(months))
    df = pd.DataFrame({"region_code": "GT", "month": months, "ndvi": seasonal + noise,
                        "composites_observed": 2})

    df.loc[df["month"] == "2010-01-01", "ndvi"] = 0.9
    df.loc[df["month"] == "2010-07-01", "ndvi"] = 0.1

    result = compute_vci(df, "2010-01-01", "2019-12-31")

    assert result["vci"].between(0, 100).all()
    assert result.loc[result["month"] == "2010-01-01", "vci"].iloc[0] == pytest.approx(100)
    assert result.loc[result["month"] == "2010-07-01", "vci"].iloc[0] == pytest.approx(0)


def test_vci_handles_zero_range_without_error():
    months = pd.date_range("2010-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"region_code": "GT", "month": months, "ndvi": 0.5,
                        "composites_observed": 2})
    result = compute_vci(df, "2010-01-01", "2011-12-31")
    assert result["vci"].isna().all()


def test_ndvi_monthly_aggregation_averages_composites():
    raw = pd.DataFrame(
        {
            "region_code": ["GT", "GT", "GT"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-17", "2020-02-02"]),
            "ndvi": [0.4, 0.6, 0.5],
        }
    )
    monthly = aggregate_to_monthly(raw)

    jan = monthly[monthly["month"] == pd.Timestamp("2020-01-01")].iloc[0]
    assert jan["ndvi"] == pytest.approx(0.5)
    assert jan["composites_observed"] == 2
