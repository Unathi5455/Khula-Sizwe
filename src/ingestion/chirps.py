"""Ingest CHIRPS monthly rainfall and reduce it to per-province means.

CHIRPS publishes Africa-wide GeoTIFFs, one per month. This script downloads
them, clips each to every South African province, and writes a tidy
region-month rainfall table.

Downloads are cached on disk, so re-running only fetches missing months.

Usage:
    python -m src.ingestion.chirps
    python -m src.ingestion.chirps --start 2015-01 --end 2024-12
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.mask import mask

from src.ingestion.boundaries import load_boundaries
from src.ingestion.utils import (
    get_logger,
    load_sources,
    request_with_retry,
    resolve_path,
)

LOG = get_logger("ingestion.chirps")


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) tuples from 'YYYY-MM' bounds."""
    periods = pd.period_range(start=start, end=end, freq="M")
    return [(p.year, p.month) for p in periods]


def download_month(
    year: int,
    month: int,
    cache_dir: Path,
    config: dict,
    session: requests.Session | None = None,
) -> Path | None:
    """Download and decompress one monthly CHIRPS raster. Returns the .tif path."""
    cfg = config["chirps"]
    tif_path = cache_dir / f"chirps-v2.0.{year}.{month:02d}.tif"
    if tif_path.exists():
        return tif_path

    filename = cfg["filename_template"].format(year=year, month=month)
    url = f"{cfg['base_url']}/{filename}"
    gz_path = cache_dir / filename

    LOG.info("Downloading %s", url)
    try:
        response = request_with_retry(
            url,
            retries=cfg["retries"],
            backoff_seconds=cfg["backoff_seconds"],
            timeout_seconds=cfg["timeout_seconds"],
            stream=True,
            session=session,
            logger=LOG,
        )
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 404:
            # Recent months are published on a lag; skip rather than crash.
            LOG.warning("%s-%02d not published yet (404), skipping", year, month)
            return None
        raise

    with gz_path.open("wb") as fh:
        for block in response.iter_content(chunk_size=1 << 20):
            fh.write(block)

    with gzip.open(gz_path, "rb") as src, tif_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_path.unlink()

    return tif_path


def zonal_mean(tif_path: Path, boundaries, nodata: float) -> list[dict]:
    """Mean rainfall per province for one raster.

    Uses the raster's own nodata value plus the configured sentinel, because
    CHIRPS marks ocean and unmapped cells with -9999 and averaging those in
    would drag every coastal province toward a nonsense negative value.
    """
    records: list[dict] = []

    with rasterio.open(tif_path) as src:
        raster_nodata = src.nodata if src.nodata is not None else nodata

        for row in boundaries.itertuples():
            clipped, _ = mask(src, [row.geometry], crop=True, filled=True,
                              nodata=raster_nodata)
            values = clipped[0].astype("float64")
            valid = values[(values != raster_nodata) & (values != nodata) & (values >= 0)]

            records.append(
                {
                    "region_code": row.region_code,
                    "region_name": row.province,
                    "rainfall_mm": float(np.mean(valid)) if valid.size else np.nan,
                    "pixel_count": int(valid.size),
                }
            )

    return records


def main() -> None:
    config = load_sources()

    parser = argparse.ArgumentParser(description="Ingest CHIRPS monthly rainfall.")
    parser.add_argument("--start", default=config["period"]["start"][:7])
    parser.add_argument("--end", default=config["period"]["end"][:7])
    parser.add_argument(
        "--keep-rasters",
        action="store_true",
        help="Keep downloaded GeoTIFFs (roughly 8 MB each) instead of deleting them.",
    )
    args = parser.parse_args()

    boundaries = load_boundaries()
    cache_dir = resolve_path(config["paths"]["raw_chirps"])
    nodata = config["chirps"]["nodata"]

    session = requests.Session()
    rows: list[dict] = []

    for year, month in month_range(args.start, args.end):
        tif_path = download_month(year, month, cache_dir, config, session=session)
        if tif_path is None:
            continue

        month_stamp = pd.Timestamp(year=year, month=month, day=1)
        for record in zonal_mean(tif_path, boundaries, nodata):
            rows.append({**record, "month": month_stamp})

        if not args.keep_rasters:
            tif_path.unlink()

        LOG.info("Processed %s-%02d", year, month)

    frame = (
        pd.DataFrame(rows)[["region_code", "region_name", "month", "rainfall_mm",
                            "pixel_count"]]
        .sort_values(["region_code", "month"])
        .reset_index(drop=True)
    )

    out_path = cache_dir / "chirps_monthly.parquet"
    frame.to_parquet(out_path, index=False)
    LOG.info("Wrote %s region-months to %s", len(frame), out_path)


if __name__ == "__main__":
    main()
