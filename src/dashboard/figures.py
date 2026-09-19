"""Pure functions that build the dashboard's figures and tables from data.

Kept separate from app.py deliberately: Dash callbacks are awkward to unit
test directly (they need a running server), but the functions that actually
build each figure don't need Dash at all - they take a DataFrame and return
a Plotly figure or a list of rows. Testing those in isolation means the
dashboard's actual logic is verified, not just "the app starts".
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.processing.spi import classify_spi

# Colour by severity, matching classify_spi's categories - red for dry,
# blue for wet, grey for unknown, so the map and table agree visually.
RISK_COLORS = {
    "extremely_dry": "#7f0000",
    "severely_dry": "#b30000",
    "moderately_dry": "#e34a33",
    "near_normal": "#fee8c8",
    "moderately_wet": "#a6bddb",
    "very_wet": "#3690c0",
    "extremely_wet": "#045a8d",
    "unknown": "#cccccc",
}

RISK_ORDER = list(RISK_COLORS.keys())


def get_latest_month(df: pd.DataFrame, date_col: str = "month") -> pd.Timestamp:
    """Most recent month with at least one non-null spi_3 value.

    Not just df[date_col].max(): the newest month or two can be all-NaN if
    SPI's rolling window hasn't filled in yet, and showing an empty map for
    "the latest month" is a worse default than showing the latest USABLE one.
    """
    usable = df.dropna(subset=["spi_3"])
    if usable.empty:
        raise ValueError("No region-months have a computed spi_3 value.")
    return usable[date_col].max()


def load_province_geojson(gpkg_path: str) -> dict:
    """Load the province boundaries as GeoJSON, keyed by region_code."""
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf.__geo_interface__


def build_choropleth(
    df: pd.DataFrame, geojson: dict, month: pd.Timestamp, spi_col: str = "spi_3"
) -> go.Figure:
    """Province-level choropleth of SPI for one month."""
    month_df = df[df["month"] == month].copy()
    if month_df.empty:
        raise ValueError(f"No rows for month {month}.")

    month_df["risk_category"] = month_df[spi_col].apply(classify_spi)

    fig = px.choropleth(
        month_df,
        geojson=geojson,
        locations="region_code",
        featureidkey="properties.region_code",
        color=spi_col,
        color_continuous_scale="RdBu",
        range_color=(-3, 3),
        hover_name="region_name",
        hover_data={"region_code": False, spi_col: ":.2f", "risk_category": True},
        labels={spi_col: "SPI-3"},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        title=f"Drought risk (SPI-3) - {month.strftime('%B %Y')}",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def build_timeseries(
    df: pd.DataFrame, region_code: str, spi_col: str = "spi_3"
) -> go.Figure:
    """SPI history for one region, with McKee drought thresholds marked."""
    region_df = df[df["region_code"] == region_code].sort_values("month")
    if region_df.empty:
        raise ValueError(f"No rows for region {region_code}.")

    region_name = region_df["region_name"].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=region_df["month"], y=region_df[spi_col],
        mode="lines", name="SPI-3", line={"color": "#2c3e50", "width": 2},
    ))

    for level, label, color in [
        (-1.0, "Moderately dry", "#e34a33"),
        (-1.5, "Severely dry", "#b30000"),
        (-2.0, "Extremely dry", "#7f0000"),
    ]:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                       annotation_text=label, annotation_position="right")

    fig.update_layout(
        title=f"{region_name} - SPI-3 over time",
        xaxis_title="Month", yaxis_title="SPI-3",
        margin={"l": 40, "r": 100, "t": 40, "b": 40},
    )
    return fig


def build_risk_table(df: pd.DataFrame, month: pd.Timestamp, spi_col: str = "spi_3") -> list[dict]:
    """One row per region for the latest month, sorted driest first."""
    month_df = df[df["month"] == month].copy()
    month_df["risk_category"] = month_df[spi_col].apply(classify_spi)
    month_df = month_df.sort_values(spi_col, ascending=True, na_position="last")

    return [
        {
            "region": row["region_name"],
            "spi_3": None if pd.isna(row[spi_col]) else round(row[spi_col], 2),
            "risk": row["risk_category"],
            "color": RISK_COLORS.get(row["risk_category"], RISK_COLORS["unknown"]),
        }
        for _, row in month_df.iterrows()
    ]


def available_months(df: pd.DataFrame, date_col: str = "month") -> list[pd.Timestamp]:
    """Sorted list of months that have at least one usable spi_3 value."""
    usable = df.dropna(subset=["spi_3"])
    return sorted(usable[date_col].unique())


def available_regions(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(region_code, region_name) pairs, sorted alphabetically by name, for a dropdown."""
    pairs = df[["region_code", "region_name"]].drop_duplicates()
    return sorted(pairs.itertuples(index=False, name=None), key=lambda p: p[1])
