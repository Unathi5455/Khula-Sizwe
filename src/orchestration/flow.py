"""Orchestrate the Khula-Sizwe pipeline: ingestion through feature building.

Each stage is the exact command-line entry point already used when running
the pipeline by hand (python -m src.ingestion.boundaries, etc.), wrapped as
a Prefect task. This keeps orchestration decoupled from pipeline logic - a
stage that works standalone works identically here, and nothing about
ingestion or processing needed to change to fit into this flow. If a stage's
CLI changes later, only its one-line wrapper here needs updating.

DEPENDENCY GRAPH
-----------------
Boundaries and NASA POWER don't depend on each other, so they run
concurrently. CHIRPS clips its rasters against the boundary file, so it must
wait for boundaries to finish first. Feature building needs both ingestion
outputs, so it waits for whichever of POWER/CHIRPS finishes last. This
mirrors exactly the order you'd run these by hand - the flow just makes the
ordering explicit and automatic instead of something to remember.

Usage:
    python -m src.orchestration.flow
    python -m src.orchestration.flow --regions GT WC FS --start 2015-01-01 --end 2024-12-31
    python -m src.orchestration.flow --with-ndvi
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Must be set before importing prefect - these silence background telemetry
# that otherwise prints a harmless but alarming-looking "database is locked"
# warning on some machines when a flow starts and stops quickly.
os.environ.setdefault("PREFECT_CLIENT_METRICS_ENABLED", "false")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from prefect import flow, get_run_logger, task  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_module(module: str, args: list[str] | None = None) -> None:
    """Run `python -m <module> [args]` as a subprocess, raising on failure.

    Subprocess rather than importing and calling main() directly: each
    ingestion/processing script already works correctly as a standalone CLI
    command (proven in Phases 1-2), so shelling out to the exact command a
    person would type keeps orchestration a thin wrapper rather than a
    second code path that could quietly drift from the tested one.
    """
    cmd = [sys.executable, "-m", module, *(args or [])]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{module} failed (exit {result.returncode}).\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


@task(name="ingest-boundaries", retries=2, retry_delay_seconds=10)
def ingest_boundaries() -> None:
    get_run_logger().info("Fetching province boundaries")
    run_module("src.ingestion.boundaries")


@task(name="ingest-nasa-power", retries=2, retry_delay_seconds=15)
def ingest_nasa_power(regions: list[str] | None, start: str, end: str) -> None:
    get_run_logger().info("Fetching NASA POWER climate data (%s to %s)", start, end)
    args = ["--start", start, "--end", end]
    if regions:
        args += ["--regions", *regions]
    run_module("src.ingestion.nasa_power", args)


@task(name="ingest-chirps", retries=1, retry_delay_seconds=30)
def ingest_chirps(start: str, end: str) -> None:
    get_run_logger().info("Fetching CHIRPS rainfall data (%s to %s)", start, end)
    run_module("src.ingestion.chirps", ["--start", start[:7], "--end", end[:7]])


@task(name="build-features", retries=1, retry_delay_seconds=10)
def build_features(skip_ndvi: bool = True) -> None:
    get_run_logger().info("Building the drought feature table")
    run_module("src.processing.build_features", ["--skip-ndvi"] if skip_ndvi else [])


@flow(name="khula-sizwe-pipeline")
def khula_sizwe_pipeline(
    regions: list[str] | None = None,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    skip_ndvi: bool = True,
) -> None:
    """Run the full ingestion-to-features pipeline end to end.

    .submit() (rather than calling tasks directly) is what makes boundaries
    and NASA POWER run concurrently instead of one blocking the other -
    calling a task directly executes it immediately and waits, while
    .submit() hands it to Prefect's task runner and returns right away.
    wait_for=[...] then creates an explicit ordering dependency even where
    there's no data passed between tasks, which is how CHIRPS is told to
    wait for boundaries despite not taking its return value as an argument.
    """
    boundaries_future = ingest_boundaries.submit()
    power_future = ingest_nasa_power.submit(regions, start, end)
    chirps_future = ingest_chirps.submit(start, end, wait_for=[boundaries_future])
    build_features.submit(skip_ndvi, wait_for=[power_future, chirps_future]).result()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Khula-Sizwe pipeline end to end.")
    parser.add_argument("--regions", nargs="*", default=None,
                         help="Region codes for NASA POWER, e.g. --regions GT WC. Default: all 9.")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--with-ndvi", action="store_true",
                         help="Also fold in NDVI/VCI (requires the manual AppEEARS export).")
    args = parser.parse_args()

    khula_sizwe_pipeline(
        regions=args.regions, start=args.start, end=args.end,
        skip_ndvi=not args.with_ndvi,
    )


if __name__ == "__main__":
    main()
