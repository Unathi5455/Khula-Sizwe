"""Tests for the ingestion layer, using synthetic data so no network is needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion.chirps import month_range
from src.ingestion.nasa_power import (
    _year_chunks,
    aggregate_to_monthly,
    parse_power_response,
)
from src.ingestion.utils import load_regions, load_sources


def _power_payload() -> dict:
    """Mimic the NASA POWER JSON shape, including a -999 fill value."""
    dates = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    keys = [d.strftime("%Y%m%d") for d in dates]
    rain = {k: 2.0 for k in keys}
    rain[keys[5]] = -999.0  # missing observation sentinel
    return {
        "properties": {
            "parameter": {
                "PRECTOTCORR": rain,
                "T2M": {k: 20.0 for k in keys},
                "GWETROOT": {k: 0.5 for k in keys},
            }
        }
    }


def test_parse_power_response_converts_fill_to_na():
    frame = parse_power_response(_power_payload(), fill_value=-999.0)
    assert len(frame) == 60
    assert frame["PRECTOTCORR"].isna().sum() == 1
    assert not (frame["PRECTOTCORR"] == -999.0).any()


def test_parse_power_response_rejects_bad_shape():
    with pytest.raises(ValueError):
        parse_power_response({"unexpected": {}}, fill_value=-999.0)


def test_aggregate_sums_rain_and_averages_temperature():
    daily = parse_power_response(_power_payload(), fill_value=-999.0)
    daily.insert(0, "region_code", "GT")
    daily.insert(1, "region_name", "Gauteng")

    monthly = aggregate_to_monthly(daily)

    assert len(monthly) == 2
    jan = monthly[monthly["month"] == pd.Timestamp("2020-01-01")].iloc[0]
    # 31 days, one missing, 2 mm each -> 60 mm summed, temperature still averaged.
    assert jan["PRECTOTCORR"] == pytest.approx(60.0)
    assert jan["T2M"] == pytest.approx(20.0)
    assert jan["observed_days"] == 31


def test_aggregate_blanks_rainfall_for_sparse_months():
    daily = parse_power_response(_power_payload(), fill_value=-999.0)
    daily.insert(0, "region_code", "GT")
    daily.insert(1, "region_name", "Gauteng")
    # Keep only 10 days of February so it falls under the coverage threshold.
    feb = daily["date"].dt.month == 2
    daily = pd.concat([daily[~feb], daily[feb].head(10)], ignore_index=True)

    monthly = aggregate_to_monthly(daily)
    feb_row = monthly[monthly["month"] == pd.Timestamp("2020-02-01")].iloc[0]

    assert feb_row["observed_days"] == 10
    assert pd.isna(feb_row["PRECTOTCORR"])
    assert feb_row["T2M"] == pytest.approx(20.0)


def test_year_chunks_cover_range_without_gaps():
    from datetime import date

    chunks = _year_chunks(date(1991, 1, 1), date(2024, 12, 31), 10)
    assert chunks[0][0] == date(1991, 1, 1)
    assert chunks[-1][1] == date(2024, 12, 31)
    for earlier, later in zip(chunks, chunks[1:]):
        assert (later[0] - earlier[1]).days == 1


def test_month_range_is_inclusive():
    months = month_range("2020-01", "2020-12")
    assert len(months) == 12
    assert months[0] == (2020, 1)
    assert months[-1] == (2020, 12)


def test_zonal_mean_excludes_nodata(tmp_path):
    """A raster half filled with -9999 must not drag the province mean negative."""
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from src.ingestion.chirps import zonal_mean

    data = np.full((10, 10), 50.0, dtype="float32")
    data[:5, :] = -9999.0
    path = tmp_path / "test.tif"

    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10, count=1,
        dtype="float32", crs="EPSG:4326",
        transform=from_origin(20.0, -20.0, 0.05, 0.05), nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    boundaries = gpd.GeoDataFrame(
        {"region_code": ["TEST"], "province": ["Test"]},
        geometry=[box(20.0, -20.5, 20.5, -20.0)],
        crs="EPSG:4326",
    )

    result = zonal_mean(path, boundaries, nodata=-9999.0)[0]
    assert result["rainfall_mm"] == pytest.approx(50.0)
    assert result["pixel_count"] == 50


def test_config_regions_are_complete_and_consistent():
    regions = load_regions()
    assert len(regions) == 9

    codes = [r["code"] for r in regions]
    assert len(set(codes)) == 9

    for region in regions:
        lat, lon = region["point"]["lat"], region["point"]["lon"]
        # Sanity-check every coordinate actually falls inside South Africa.
        assert -35.0 <= lat <= -22.0, f"{region['code']} latitude out of range"
        assert 16.0 <= lon <= 33.0, f"{region['code']} longitude out of range"
        assert region["rainfall_regime"] in {"summer", "winter", "year_round"}


def test_sources_config_has_required_keys():
    config = load_sources()
    assert "PRECTOTCORR" in config["nasa_power"]["parameters"]
    assert "GWETROOT" in config["nasa_power"]["parameters"]
    assert config["period"]["baseline_start"] < config["period"]["baseline_end"]