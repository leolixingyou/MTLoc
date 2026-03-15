# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import time
from http.client import responses
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import urllib3

from .. import logger
from ..utils.geo import BoundaryBox

OSM_URL = "https://api.openstreetmap.org/api/0.6/map.json"


def _fetch_osm_chunk(left, bottom, right, top, retries=3, timeout=30):
    """Fetch a single OSM chunk with retries."""
    query = {"bbox": f"{left},{bottom},{right},{top}"}
    for attempt in range(retries):
        try:
            result = urllib3.request("GET", OSM_URL, fields=query, timeout=timeout)
            if result.status == 200:
                return result.json()
            error = result.data.decode("utf-8", errors="replace")[:200]
            logger.warning(
                "OSM API returned %d on attempt %d: %s", result.status, attempt + 1, error
            )
        except Exception as e:
            logger.warning("OSM request failed attempt %d: %s", attempt + 1, e)
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise ValueError(f"Failed to fetch OSM data after {retries} attempts")


def _merge_osm_results(results):
    """Merge multiple OSM JSON responses, deduplicating by element ID."""
    seen_ids = set()
    merged_elements = []
    merged_bounds = None

    for data in results:
        bounds = data.get("bounds")
        if bounds is not None:
            if merged_bounds is None:
                merged_bounds = dict(bounds)
            else:
                merged_bounds["minlat"] = min(merged_bounds["minlat"], bounds["minlat"])
                merged_bounds["minlon"] = min(merged_bounds["minlon"], bounds["minlon"])
                merged_bounds["maxlat"] = max(merged_bounds["maxlat"], bounds["maxlat"])
                merged_bounds["maxlon"] = max(merged_bounds["maxlon"], bounds["maxlon"])

        for elem in data.get("elements", []):
            key = (elem["type"], elem["id"])
            if key not in seen_ids:
                seen_ids.add(key)
                merged_elements.append(elem)

    result = {"version": "0.6", "elements": merged_elements}
    if merged_bounds is not None:
        result["bounds"] = merged_bounds
    return result


def get_osm(
    boundary_box: BoundaryBox,
    cache_path: Optional[Path] = None,
    overwrite: bool = False,
    max_chunk_deg: float = 0.01,
) -> Dict[str, Any]:
    if not overwrite and cache_path is not None and cache_path.is_file():
        return json.loads(cache_path.read_text())

    (bottom, left), (top, right) = boundary_box.min_, boundary_box.max_
    lat_span = top - bottom
    lon_span = right - left

    # If bbox is small enough, fetch directly
    if lat_span <= max_chunk_deg and lon_span <= max_chunk_deg:
        logger.info("Calling the OpenStreetMap API (single request)...")
        data = _fetch_osm_chunk(left, bottom, right, top)
        if cache_path is not None:
            cache_path.write_bytes(json.dumps(data).encode())
        return data

    # Split into chunks for large areas
    n_lat = max(1, int(np.ceil(lat_span / max_chunk_deg)))
    n_lon = max(1, int(np.ceil(lon_span / max_chunk_deg)))
    total = n_lat * n_lon
    logger.info(
        "Large area: splitting into %d x %d = %d chunks (%.4f° x %.4f°)",
        n_lat, n_lon, total, lat_span / n_lat, lon_span / n_lon,
    )

    lat_edges = np.linspace(bottom, top, n_lat + 1)
    lon_edges = np.linspace(left, right, n_lon + 1)

    results = []
    for i in range(n_lat):
        for j in range(n_lon):
            chunk_idx = i * n_lon + j + 1
            logger.info("  Fetching chunk %d/%d...", chunk_idx, total)
            data = _fetch_osm_chunk(
                lon_edges[j], lat_edges[i], lon_edges[j + 1], lat_edges[i + 1]
            )
            results.append(data)
            if chunk_idx < total:
                time.sleep(0.5)  # Rate limiting

    merged = _merge_osm_results(results)
    logger.info("Merged: %d unique elements from %d chunks", len(merged["elements"]), total)

    if cache_path is not None:
        cache_path.write_bytes(json.dumps(merged).encode())
    return merged
