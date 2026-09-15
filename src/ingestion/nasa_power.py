"""Ingest daily climate data from the NASA POWER point API, per province.

Pulls daily values (more flexible downstream than the monthly endpoint), writes
the raw daily frame to Parquet, and also writes a monthly aggregate ready for
the processing layer.

Usage:
    python -m src.ingestion.nasa_power
    python -m src.ingestion.nasa_power --regions GT WC --start 2015-01-01
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd
import requests

from src.ingestion.utils import (
    get_logger,
    load_regions,
    load_sources,
    request_with_retry,
    resolve_path,
)

LOG = get_logger("ingestion.nasa_power")

# Daily variables that should be summed over a month rather than averaged.
SUM_PARAMETERS = {"PRECTOTCORR"}


def _year_chunks(start: date, end: date, chunk_years: int) -> list[tuple[date, date]]:
    """Split a date range into chunks so requests stay within API limits."""
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(date(cursor.year + chunk_years - 1, 12, 31), end)
        chunks.append((cursor, chunk_end))
        cursor = date(chunk_end.year + 1, 1, 1)
    return chunks


def fetch_region_daily(
    region: dict,
    start: date,
    end: date,
    config: dict,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch the full daily series for one region, chunked by year range."""
    cfg = config["nasa_power"]
    frames: list[pd.DataFrame] = []

    for chunk_start, chunk_end in _year_chunks(start, end, cfg["chunk_years"]):
        LOG.info(
            "Fetching %s (%s) %s to %s",
            region["name"],
            region["code"],
            chunk_start,
            chunk_end,
        )
        params = {
            "parameters": ",".join(cfg["parameters"]),
            "community": cfg["community"],
            "latitude": region["point"]["lat"],
            "longitude": region["point"]["lon"],
            "start": chunk_start.strftime("%Y%m%d"),
            "end": chunk_end.strftime("%Y%m%d"),
            "format": "JSON",
        }
        response = request_with_retry(
            cfg["base_url"],
            params=params,
            retries=cfg["retries"],
            backoff_seconds=cfg["backoff_seconds"],
            timeout_seconds=cfg["timeout_seconds"],
            session=session,
            logger=LOG,
        )
        frames.append(parse_power_response(response.json(), cfg["fill_value"]))

    daily = pd.concat(frames, ignore_index=True)
    daily.insert(0, "region_code", region["code"])
    daily.insert(1, "region_name", region["name"])
    return daily.sort_values("date").reset_index(drop=True)


def parse_power_response(payload: dict, fill_value: float) -> pd.DataFrame:
    """Turn the POWER JSON payload into a tidy daily DataFrame.

    POWER returns {"properties": {"parameter": {"T2M": {"20150101": 21.4, ...}}}}
    and uses a sentinel fill value (-999) for missing observations, which must
    become NaN or it will silently poison every downstream average.
    """
    try:
        parameters = payload["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unexpected NASA POWER response shape: {exc}") from exc

    frame = pd.DataFrame(parameters)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame = frame.replace(fill_value, pd.NA).astype("Float64")
    frame.index.name = "date"
    return frame.reset_index()


def aggregate_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Collapse daily values to one row per region-month.

    Precipitation is summed (a month's total rainfall); everything else is
    averaged. `observed_days` is kept so the processing layer can discard
    months with too little coverage rather than trusting a partial average.
    """
    value_columns = [
        col
        for col in daily.columns
        if col not in {"region_code", "region_name", "date"}
    ]
    agg_map = {
        col: ("sum" if col in SUM_PARAMETERS else "mean") for col in value_columns
    }

    working = daily.copy()
    working["month"] = working["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        working.groupby(["region_code", "region_name", "month"], as_index=False)
        .agg({**agg_map, "date": "count"})
        .rename(columns={"date": "observed_days"})
    )

    # A summed rainfall total over a half-reported month is misleading, so blank
    # it out rather than letting it look like a genuinely dry month.
    sparse = monthly["observed_days"] < 20
    if sparse.any():
        LOG.warning("%s region-months have under 20 observed days", int(sparse.sum()))
        for col in SUM_PARAMETERS & set(monthly.columns):
            monthly.loc[sparse, col] = pd.NA

    return monthly.sort_values(["region_code", "month"]).reset_index(drop=True)


def main() -> None:
    config = load_sources()
    all_regions = load_regions()

    parser = argparse.ArgumentParser(description="Ingest NASA POWER climate data.")
    parser.add_argument(
        "--regions",
        nargs="*",
        default=None,
        help="Region codes to fetch (default: all). E.g. --regions GT WC",
    )
    parser.add_argument("--start", default=config["period"]["start"])
    parser.add_argument("--end", default=config["period"]["end"])
    args = parser.parse_args()

    regions = all_regions
    if args.regions:
        wanted = {code.upper() for code in args.regions}
        regions = [r for r in all_regions if r["code"] in wanted]
        missing = wanted - {r["code"] for r in regions}
        if missing:
            raise SystemExit(f"Unknown region codes: {sorted(missing)}")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_dir = resolve_path(config["paths"]["raw_nasa_power"])

    session = requests.Session()
    daily_frames: list[pd.DataFrame] = []

    for region in regions:
        daily = fetch_region_daily(region, start, end, config, session=session)
        region_path = out_dir / f"{region['code']}_daily.parquet"
        daily.to_parquet(region_path, index=False)
        LOG.info("Wrote %s rows to %s", len(daily), region_path)
        daily_frames.append(daily)

    combined = pd.concat(daily_frames, ignore_index=True)
    monthly = aggregate_to_monthly(combined)

    monthly_path = out_dir / "nasa_power_monthly.parquet"
    monthly.to_parquet(monthly_path, index=False)
    LOG.info(
        "Wrote %s region-months (%s regions) to %s",
        len(monthly),
        monthly["region_code"].nunique(),
        monthly_path,
    )


if __name__ == "__main__":
    main()