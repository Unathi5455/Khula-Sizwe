"""Tests for the dashboard's figure-building functions.

These test the pure functions in figures.py directly, not the Dash app
itself - a running server isn't needed to verify that a choropleth gets the
right region colored, or that the risk table is sorted driest-first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.dashboard.figures import (
    available_months,
    available_regions,
    build_choropleth,
    build_risk_table,
    build_timeseries,
    get_latest_month,
)


def _toy_features() -> pd.DataFrame:
    months = pd.date_range("2020-01-01", periods=4, freq="MS")
    rows = []
    # Region A: a clear drought signal in the most recent month.
    for m, spi in zip(months, [0.2, -0.5, -1.2, -2.1]):
        rows.append({"region_code": "A", "region_name": "Region A", "month": m, "spi_3": spi})
    # Region B: stays wet throughout, and its LAST month is null (simulates
    # SPI's rolling-window edge effect) to check get_latest_month skips it.
    for m, spi in zip(months, [1.0, 1.5, 2.0, np.nan]):
        rows.append({"region_code": "B", "region_name": "Region B", "month": m, "spi_3": spi})
    return pd.DataFrame(rows)


def _toy_geojson() -> dict:
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(
        {"region_code": ["A", "B"], "province": ["Region A", "Region B"]},
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)],
        crs="EPSG:4326",
    )
    return gdf.__geo_interface__


def test_get_latest_month_skips_all_null_tail():
    df = _toy_features()
    latest = get_latest_month(df)
    assert latest == pd.Timestamp("2020-04-01")


def test_get_latest_month_raises_when_nothing_usable():
    df = pd.DataFrame({"region_code": ["A"], "month": [pd.Timestamp("2020-01-01")], "spi_3": [np.nan]})
    with pytest.raises(ValueError, match="No region-months"):
        get_latest_month(df)


def test_available_months_excludes_fully_null_months():
    df = _toy_features()
    months = available_months(df)
    assert len(months) == 4


def test_available_regions_sorted_by_name():
    df = _toy_features()
    regions = available_regions(df)
    assert regions == [("A", "Region A"), ("B", "Region B")]


def test_build_choropleth_assigns_correct_risk_category():
    df = _toy_features()
    geojson = _toy_geojson()
    fig = build_choropleth(df, geojson, pd.Timestamp("2020-04-01"))

    # customdata carries [region_code, spi_3, risk_category] in that order
    # (region_code stays in the array even though hover_data marks it False
    # - that only hides it from the hover text). Region A at spi=-2.1
    # should be extremely_dry.
    trace = fig.data[0]
    locations = list(trace.locations)
    a_index = locations.index("A")
    risk_categories = [row[2] for row in trace.customdata]
    assert risk_categories[a_index] == "extremely_dry"


def test_build_choropleth_raises_for_missing_month():
    df = _toy_features()
    geojson = _toy_geojson()
    with pytest.raises(ValueError, match="No rows for month"):
        build_choropleth(df, geojson, pd.Timestamp("2099-01-01"))


def test_build_timeseries_includes_full_history_for_region():
    df = _toy_features()
    fig = build_timeseries(df, "A")
    spi_trace = fig.data[0]
    assert list(spi_trace.y) == [0.2, -0.5, -1.2, -2.1]


def test_build_timeseries_raises_for_unknown_region():
    df = _toy_features()
    with pytest.raises(ValueError, match="No rows for region"):
        build_timeseries(df, "ZZZ")


def test_build_risk_table_sorted_driest_first():
    df = _toy_features()
    table = build_risk_table(df, pd.Timestamp("2020-04-01"))

    assert table[0]["region"] == "Region A"
    assert table[0]["risk"] == "extremely_dry"
    assert table[-1]["region"] == "Region B"
    assert table[-1]["spi_3"] is None


def test_build_risk_table_rounds_spi_to_two_decimals():
    df = pd.DataFrame({
        "region_code": ["A"], "region_name": ["Region A"],
        "month": [pd.Timestamp("2020-01-01")], "spi_3": [-1.23456],
    })
    table = build_risk_table(df, pd.Timestamp("2020-01-01"))
    assert table[0]["spi_3"] == -1.23
