"""Build the region-month feature table used for modeling."""

from __future__ import annotations

import argparse

import pandas as pd

from src.ingestion.utils import get_logger, load_sources, resolve_path
from src.processing.climatology import apply_anomaly, compute_climatology
from src.processing.ndvi import build_vci_table
from src.processing.spi import compute_spi

LOG = get_logger("processing.build_features")

SPI_WINDOWS = (1, 3, 6)


def load_raw_tables(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    chirps_path = (
        resolve_path(config["paths"]["raw_chirps"]) / "chirps_monthly.parquet"
    )
    power_path = (
        resolve_path(config["paths"]["raw_nasa_power"]) / "nasa_power_monthly.parquet"
    )

    for path, source in [(chirps_path, "CHIRPS"), (power_path, "NASA POWER")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{source} output not found at {path}. Run Phase 1 ingestion first."
            )

    chirps = pd.read_parquet(chirps_path)
    power = pd.read_parquet(power_path)
    return chirps, power


def merge_sources(chirps: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    power_climate = power.drop(columns=["region_name", "PRECTOTCORR"])
    merged = chirps.merge(power_climate, on=["region_code", "month"], how="inner")

    dropped = len(chirps) + len(power) - 2 * len(merged)
    if dropped:
        LOG.warning(
            "%s region-months present in only one source and dropped by the "
            "inner join.",
            abs(dropped),
        )

    return merged.sort_values(["region_code", "month"]).reset_index(drop=True)


def add_rainfall_and_soil_anomalies(
    df: pd.DataFrame, baseline_start: str, baseline_end: str
) -> pd.DataFrame:
    working = df.copy()

    for col in ["rainfall_mm", "GWETROOT", "GWETTOP", "T2M"]:
        clim = compute_climatology(working, col, baseline_start, baseline_end)
        working = apply_anomaly(working, clim, col, method="zscore")

    return working


def add_spi(df: pd.DataFrame, baseline_start: str, baseline_end: str) -> pd.DataFrame:
    working = df
    for window in SPI_WINDOWS:
        working = compute_spi(working, window, baseline_start, baseline_end)
        n_valid = working[f"spi_{window}"].notna().sum()
        LOG.info("SPI-%s computed for %s/%s rows", window, n_valid, len(working))
    return working


def add_ndvi(df: pd.DataFrame, baseline_start: str, baseline_end: str) -> pd.DataFrame:
    try:
        vci = build_vci_table(baseline_start, baseline_end)
    except FileNotFoundError as exc:
        LOG.warning("Skipping NDVI/VCI: %s", exc)
        return df

    vci_slim = vci[["region_code", "month", "ndvi", "vci"]]
    merged = df.merge(vci_slim, on=["region_code", "month"], how="left")
    return merged


def main() -> None:
    config = load_sources()
    baseline_start = config["period"]["baseline_start"]
    baseline_end = config["period"]["baseline_end"]

    parser = argparse.ArgumentParser(description="Build the drought feature table.")
    parser.add_argument("--skip-ndvi", action="store_true")
    args = parser.parse_args()

    chirps, power = load_raw_tables(config)
    merged = merge_sources(chirps, power)
    LOG.info("Merged CHIRPS + POWER: %s region-months", len(merged))

    with_anomalies = add_rainfall_and_soil_anomalies(merged, baseline_start, baseline_end)
    with_spi = add_spi(with_anomalies, baseline_start, baseline_end)

    final = with_spi if args.skip_ndvi else add_ndvi(with_spi, baseline_start, baseline_end)

    out_dir = resolve_path(config["paths"]["processed"])
    out_path = out_dir / "features.parquet"
    final.to_parquet(out_path, index=False)
    LOG.info("Wrote %s rows, %s columns to %s", len(final), final.shape[1], out_path)


if __name__ == "__main__":
    main()
