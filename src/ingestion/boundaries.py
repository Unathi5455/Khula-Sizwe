"""Download South African province boundaries (GADM level 1).

Every other spatial source gets clipped against this file, so it is the first
thing to run. Saved as GeoPackage rather than shapefile to avoid the 10-char
column name truncation that shapefiles impose.

Usage:
    python -m src.ingestion.boundaries
"""

from __future__ import annotations

import geopandas as gpd

from src.ingestion.utils import (
    get_logger,
    load_regions,
    load_sources,
    request_with_retry,
    resolve_path,
)

LOG = get_logger("ingestion.boundaries")


def download_boundaries(force: bool = False) -> gpd.GeoDataFrame:
    config = load_sources()
    cfg = config["boundaries"]
    out_dir = resolve_path(config["paths"]["raw_boundaries"])
    geojson_path = out_dir / "gadm41_ZAF_1.json"
    gpkg_path = out_dir / "za_provinces.gpkg"

    if not geojson_path.exists() or force:
        LOG.info("Downloading provinces from %s", cfg["gadm_url"])
        response = request_with_retry(cfg["gadm_url"], timeout_seconds=180, logger=LOG)
        geojson_path.write_bytes(response.content)
        LOG.info("Saved %s", geojson_path)
    else:
        LOG.info("Using cached %s", geojson_path)

    gdf = gpd.read_file(geojson_path)
    name_field = cfg["name_field"]

    if name_field not in gdf.columns:
        raise ValueError(
            f"Expected field '{name_field}' not in boundary file. "
            f"Available: {list(gdf.columns)}"
        )

    gdf = gdf[[name_field, "geometry"]].rename(columns={name_field: "province"})

    # Attach our own region codes so joins downstream key on a stable short code
    # rather than on province name strings, which differ between sources.
    lookup = {r["gadm_name"]: r["code"] for r in load_regions()}
    gdf["region_code"] = gdf["province"].map(lookup)

    unmatched = gdf[gdf["region_code"].isna()]["province"].tolist()
    if unmatched:
        raise ValueError(
            "These GADM province names are not in config/regions.yml "
            f"(fix gadm_name there): {unmatched}"
        )

    # CHIRPS is published in WGS84; keep boundaries in the same CRS to clip.
    gdf = gdf.to_crs("EPSG:4326")

    gdf.to_file(gpkg_path, driver="GPKG")
    LOG.info("Wrote %s provinces to %s", len(gdf), gpkg_path)
    return gdf


def load_boundaries() -> gpd.GeoDataFrame:
    """Load the cached boundary file, downloading it if absent."""
    config = load_sources()
    gpkg_path = resolve_path(config["paths"]["raw_boundaries"]) / "za_provinces.gpkg"
    if not gpkg_path.exists():
        return download_boundaries()
    return gpd.read_file(gpkg_path)


if __name__ == "__main__":
    download_boundaries()