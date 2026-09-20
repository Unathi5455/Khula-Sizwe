"""Tests for the orchestration flow.

These mock run_module (the one function that actually shells out) so tests
run in milliseconds against the REAL flow and task wiring - the same
dependency graph, retry config, and argument-building code that runs for
real, just without waiting on real downloads.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.orchestration import flow as flow_module


def test_stages_run_in_correct_dependency_order():
    """Boundaries and NASA POWER should start almost simultaneously (neither
    depends on the other); CHIRPS must not start until boundaries finishes;
    features must not start until both ingestion stages finish.
    """
    call_log: list[str] = []

    def fake_run_module(module, args=None):
        if module == "src.ingestion.boundaries":
            time.sleep(0.25)  # deliberately the slowest, to prove real waiting
        call_log.append(module)

    with patch.object(flow_module, "run_module", side_effect=fake_run_module):
        flow_module.khula_sizwe_pipeline(regions=["GT"], start="2020-01-01", end="2020-02-01")

    assert call_log.index("src.ingestion.boundaries") < call_log.index("src.ingestion.chirps")
    assert call_log.index("src.ingestion.nasa_power") < call_log.index("src.processing.build_features")
    assert call_log.index("src.ingestion.chirps") < call_log.index("src.processing.build_features")


def test_boundaries_and_power_run_concurrently_not_sequentially():
    """Record each task's start/end time and check their execution windows
    overlap. A wall-clock total-time threshold would be flaky here - Prefect
    has a few seconds of fixed per-run server bootstrap overhead that has
    nothing to do with the tasks themselves and varies by machine - so this
    checks the actual structural property instead: if power starts before
    boundaries finishes, they genuinely ran concurrently, regardless of how
    much fixed overhead surrounds them.
    """
    windows: dict[str, list[float]] = {}

    def fake_run_module(module, args=None):
        windows.setdefault(module, []).append(time.monotonic())
        if module == "src.ingestion.boundaries":
            time.sleep(0.3)
        elif module == "src.ingestion.nasa_power":
            time.sleep(0.3)
        windows[module].append(time.monotonic())

    with patch.object(flow_module, "run_module", side_effect=fake_run_module):
        flow_module.khula_sizwe_pipeline(regions=["GT"], start="2020-01-01", end="2020-02-01")

    boundaries_start, boundaries_end = windows["src.ingestion.boundaries"]
    power_start, power_end = windows["src.ingestion.nasa_power"]

    overlap = power_start < boundaries_end and boundaries_start < power_end
    assert overlap, (
        f"Expected overlapping execution windows (concurrent), got "
        f"boundaries=[{boundaries_start:.2f}, {boundaries_end:.2f}] "
        f"power=[{power_start:.2f}, {power_end:.2f}] (no overlap = ran sequentially)"
    )


def test_correct_arguments_passed_to_each_stage():
    call_log: list[tuple[str, list | None]] = []

    def fake_run_module(module, args=None):
        call_log.append((module, args))

    with patch.object(flow_module, "run_module", side_effect=fake_run_module):
        flow_module.khula_sizwe_pipeline(
            regions=["GT", "WC"], start="2020-01-01", end="2020-03-01", skip_ndvi=True,
        )

    calls = dict(call_log)
    assert calls["src.ingestion.nasa_power"] == [
        "--start", "2020-01-01", "--end", "2020-03-01", "--regions", "GT", "WC",
    ]
    assert calls["src.ingestion.chirps"] == ["--start", "2020-01", "--end", "2020-03"]
    assert calls["src.processing.build_features"] == ["--skip-ndvi"]
    assert calls["src.ingestion.boundaries"] is None


def test_with_ndvi_flag_changes_build_features_args():
    call_log: list[tuple[str, list | None]] = []

    def fake_run_module(module, args=None):
        call_log.append((module, args))

    with patch.object(flow_module, "run_module", side_effect=fake_run_module):
        flow_module.khula_sizwe_pipeline(
            regions=["GT"], start="2020-01-01", end="2020-02-01", skip_ndvi=False,
        )

    calls = dict(call_log)
    assert calls["src.processing.build_features"] == []


def test_no_regions_means_no_regions_flag_passed():
    """Omitting --regions is how the underlying script knows to default to
    all 9 provinces - passing an empty flag would break that default."""
    call_log: list[tuple[str, list | None]] = []

    def fake_run_module(module, args=None):
        call_log.append((module, args))

    with patch.object(flow_module, "run_module", side_effect=fake_run_module):
        flow_module.khula_sizwe_pipeline(regions=None, start="2020-01-01", end="2020-02-01")

    calls = dict(call_log)
    assert "--regions" not in calls["src.ingestion.nasa_power"]


def test_failure_in_boundaries_prevents_downstream_stages_from_running():
    """The whole point of explicit orchestration: a broken upstream stage
    must not let the pipeline silently continue and produce a feature table
    built on missing or partial data.
    """
    # retries=0 here only to keep the test fast - production keeps its real
    # retry config; this swaps in a zero-retry variant for this test only.
    fast_boundaries = flow_module.ingest_boundaries.with_options(retries=0, retry_delay_seconds=0)
    call_log: list[str] = []

    def fake_run_module(module, args=None):
        call_log.append(module)
        if module == "src.ingestion.boundaries":
            raise RuntimeError("simulated permanent failure")

    with patch.object(flow_module, "ingest_boundaries", fast_boundaries):
        with patch.object(flow_module, "run_module", side_effect=fake_run_module):
            with pytest.raises(Exception):
                flow_module.khula_sizwe_pipeline(regions=["GT"], start="2020-01-01", end="2020-02-01")

    assert "src.ingestion.chirps" not in call_log
    assert "src.processing.build_features" not in call_log


def test_run_module_raises_on_nonzero_exit_code():
    """run_module is the one place a subprocess failure gets turned into a
    Python exception Prefect can see and retry on - if this didn't raise,
    every failure downstream of it would look like a silent success.
    """
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = "some output"
        mock_run.return_value.stderr = "some error"

        with pytest.raises(RuntimeError, match="failed"):
            flow_module.run_module("src.ingestion.boundaries")


def test_run_module_succeeds_silently_on_zero_exit_code():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        flow_module.run_module("src.ingestion.boundaries")  # should not raise
