"""Shared helpers for the ingestion layer."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_regions() -> list[dict[str, Any]]:
    return load_yaml("regions.yml")["regions"]


def load_sources() -> dict[str, Any]:
    return load_yaml("sources.yml")


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path against the project root and create it."""
    path = PROJECT_ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    return path


def request_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    retries: int = 4,
    backoff_seconds: float = 3.0,
    timeout_seconds: float = 60.0,
    stream: bool = False,
    session: requests.Session | None = None,
    logger: logging.Logger | None = None,
) -> requests.Response:
    """GET with exponential backoff.

    Retries on connection errors, timeouts, and 5xx/429 responses. Raises on
    4xx other than 429, since those signal a bad request rather than a blip.
    """
    log = logger or get_logger("ingestion.http")
    caller = session or requests
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = caller.get(
                url, params=params, timeout=timeout_seconds, stream=stream
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(
                    f"Retryable status {response.status_code}", response=response
                )
            response.raise_for_status()
            return response
        except (requests.RequestException, requests.HTTPError) as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                raise
            if attempt == retries:
                break
            wait = backoff_seconds * (2 ** (attempt - 1))
            log.warning(
                "Request failed (attempt %s/%s): %s. Retrying in %.1fs",
                attempt,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"Request to {url} failed after {retries} attempts") from last_error