"""Khula-Sizwe drought early-warning dashboard.

Reads Phase 2's feature table directly - no trained model required, since
SPI on its own is already a usable (if simple) risk signal. Shows a
province-level map for a selected month, a per-region time series with
drought thresholds marked, and a sortable risk table.

Usage:
    python -m src.dashboard.app
Then open http://127.0.0.1:8050 in a browser.
"""

from __future__ import annotations

from dash import Dash, dcc, html, Input, Output
import pandas as pd

from src.ingestion.utils import get_logger, load_sources, resolve_path
from src.dashboard.figures import (
    available_months,
    available_regions,
    build_choropleth,
    build_risk_table,
    build_timeseries,
    get_latest_month,
    load_province_geojson,
)

LOG = get_logger("dashboard.app")


def load_data() -> tuple[pd.DataFrame, dict]:
    config = load_sources()
    features_path = resolve_path(config["paths"]["processed"]) / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"No feature table at {features_path}. Run Phase 2 "
            "(python -m src.processing.build_features) first."
        )
    df = pd.read_parquet(features_path)

    boundaries_path = resolve_path(config["paths"]["raw_boundaries"]) / "za_provinces.gpkg"
    if not boundaries_path.exists():
        raise FileNotFoundError(
            f"No province boundaries at {boundaries_path}. Run "
            "python -m src.ingestion.boundaries first."
        )
    geojson = load_province_geojson(str(boundaries_path))

    return df, geojson


def build_app(df: pd.DataFrame, geojson: dict) -> Dash:
    app = Dash(__name__, title="Khula-Sizwe - Drought Early Warning")

    months = available_months(df)
    regions = available_regions(df)
    latest = get_latest_month(df)

    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "maxWidth": "1100px", "margin": "0 auto", "padding": "20px"},
        children=[
            html.H1("Khula-Sizwe: Drought Early Warning", style={"marginBottom": "4px"}),
            html.P(
                "Province-level drought risk for South Africa, based on the Standardized "
                "Precipitation Index (SPI-3).",
                style={"color": "#555", "marginTop": "0"},
            ),

            html.Div(
                style={"display": "flex", "gap": "24px", "flexWrap": "wrap", "marginTop": "20px"},
                children=[
                    html.Div(
                        style={"flex": "2", "minWidth": "420px"},
                        children=[
                            html.Label("Month"),
                            dcc.Slider(
                                id="month-slider",
                                min=0, max=len(months) - 1, step=1,
                                value=len(months) - 1,
                                marks={
                                    i: m.strftime("%Y") for i, m in enumerate(months)
                                    if m.month == 1
                                },
                                tooltip={"placement": "bottom"},
                            ),
                            dcc.Graph(id="choropleth"),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1", "minWidth": "300px"},
                        children=[
                            html.Label("Current risk by province"),
                            html.Div(id="risk-table", style={"marginTop": "8px"}),
                        ],
                    ),
                ],
            ),

            html.Hr(style={"margin": "30px 0"}),

            html.Label("Region"),
            dcc.Dropdown(
                id="region-dropdown",
                options=[{"label": name, "value": code} for code, name in regions],
                value=regions[0][0],
                clearable=False,
                style={"maxWidth": "400px"},
            ),
            dcc.Graph(id="timeseries"),
        ],
    )

    @app.callback(Output("choropleth", "figure"), Input("month-slider", "value"))
    def update_choropleth(month_index: int):
        month = months[month_index]
        return build_choropleth(df, geojson, month)

    @app.callback(Output("risk-table", "children"), Input("month-slider", "value"))
    def update_risk_table(month_index: int):
        month = months[month_index]
        rows = build_risk_table(df, month)
        return html.Table(
            style={"width": "100%", "borderCollapse": "collapse"},
            children=[
                html.Thead(html.Tr([
                    html.Th("Province", style={"textAlign": "left", "padding": "4px"}),
                    html.Th("SPI-3", style={"textAlign": "right", "padding": "4px"}),
                    html.Th("Risk", style={"textAlign": "left", "padding": "4px"}),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(row["region"], style={"padding": "4px"}),
                        html.Td(
                            "-" if row["spi_3"] is None else f"{row['spi_3']:.2f}",
                            style={"textAlign": "right", "padding": "4px"},
                        ),
                        html.Td(
                            row["risk"].replace("_", " ").title(),
                            style={
                                "padding": "4px", "color": "#fff",
                                "backgroundColor": row["color"],
                                "borderRadius": "4px", "textAlign": "center",
                            },
                        ),
                    ])
                    for row in rows
                ]),
            ],
        )

    @app.callback(Output("timeseries", "figure"), Input("region-dropdown", "value"))
    def update_timeseries(region_code: str):
        return build_timeseries(df, region_code)

    return app


def main() -> None:
    df, geojson = load_data()
    app = build_app(df, geojson)
    LOG.info("Starting dashboard at http://127.0.0.1:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
