# Copyright (c) Li Xingyou, 2026.
# Prepare nuScenes data for OrienterNet evaluation:
#   1. Create Projection for each nuScenes location
#   2. Download OSM data for each location
#   3. Generate rasterized tile managers
#   4. Save tiles.pkl per location

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from ... import logger
from ...osm.tiling import TileManager
from ...utils.geo import BoundaryBox, Projection

# nuScenes map origins (southwest corner) from nuscenes-devkit map_api.py
# These are the GPS lat/lon reference points for each map
MAP_ORIGINS = {
    "boston-seaport": (42.336849169438615, -71.05785369873047),
    "singapore-onenorth": (1.2882100868743724, 103.78475189208984),
    "singapore-hollandvillage": (1.2993652317780957, 103.78217697143555),
    "singapore-queenstown": (1.2782562240223188, 103.76741409301758),
}


def get_ego_poses_by_location(nusc):
    """Gather all ego_pose (x, y) per location from nuScenes."""
    location_poses = defaultdict(list)
    for scene in nusc.scene:
        log = nusc.get("log", scene["log_token"])
        location = log["location"]
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            location_poses[location].append(ep["translation"][:2])
            sample_token = sample["next"]
    return {loc: np.array(poses) for loc, poses in location_poses.items()}


def prepare_osm_for_location(
    location,
    ego_poses_xy,
    output_dir,
    osm_cache_dir=None,
    tile_margin=512,
    ppm=2,
):
    """Create TileManager for one nuScenes location.

    Args:
        location: nuScenes location name (e.g. 'boston-seaport')
        ego_poses_xy: Nx2 array of ego_pose (x, y) in nuScenes local meters
        output_dir: directory to save tiles.pkl
        osm_cache_dir: directory to cache OSM data files
        tile_margin: margin (meters) around ego trajectory for map coverage
        ppm: pixels per meter for rasterization
    """
    lat_origin, lon_origin = MAP_ORIGINS[location]

    # Create Projection centered at the nuScenes map origin
    projection = Projection(lat_origin, lon_origin)

    # nuScenes ego_pose is in local meters from map origin
    # OrienterNet's Projection.project() converts GPS -> local meters
    # We need to verify coordinate alignment.
    #
    # nuScenes: x = right (east-ish), y = forward (north-ish)
    # Projection.project() returns topocentric: x = east, y = north
    # These should be compatible. The map origin is the reference point.

    # Compute bounding box in local meters around all ego poses
    bbox_meters = BoundaryBox(
        ego_poses_xy.min(0) - tile_margin,
        ego_poses_xy.max(0) + tile_margin,
    )

    logger.info(
        "Location %s: %d poses, bbox %.0f x %.0f meters",
        location,
        len(ego_poses_xy),
        *bbox_meters.size,
    )

    # Convert bbox to GPS for OSM download
    bbox_gps = projection.unproject(bbox_meters)
    logger.info(
        "GPS bbox: lat [%.6f, %.6f], lon [%.6f, %.6f]",
        bbox_gps.min_[0],
        bbox_gps.max_[0],
        bbox_gps.min_[1],
        bbox_gps.max_[1],
    )

    # Download OSM and create tiles
    osm_path = None
    if osm_cache_dir is not None:
        osm_cache_dir = Path(osm_cache_dir)
        osm_cache_dir.mkdir(parents=True, exist_ok=True)
        osm_path = osm_cache_dir / f"{location}.json"

    tile_manager = TileManager.from_bbox(
        projection,
        bbox_meters,
        ppm,
        path=osm_path,
    )

    # Save
    output_path = Path(output_dir) / f"tiles_{location}.pkl"
    tile_manager.save(output_path)
    logger.info("Saved tiles to %s", output_path)

    return tile_manager


def prepare_all(
    nusc_dataroot,
    output_dir,
    version="v1.0-mini",
    osm_cache_dir=None,
    tile_margin=512,
    ppm=2,
):
    """Prepare OSM tiles for all nuScenes locations."""
    from nuscenes.nuscenes import NuScenes

    logger.info("Loading nuScenes %s from %s", version, nusc_dataroot)
    nusc = NuScenes(version=version, dataroot=nusc_dataroot, verbose=True)

    location_poses = get_ego_poses_by_location(nusc)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_managers = {}
    for location, poses in location_poses.items():
        logger.info("Processing %s (%d poses)...", location, len(poses))
        tm = prepare_osm_for_location(
            location,
            poses,
            output_dir,
            osm_cache_dir=osm_cache_dir,
            tile_margin=tile_margin,
            ppm=ppm,
        )
        tile_managers[location] = tm

    logger.info("Done. Created tiles for %d locations.", len(tile_managers))
    return tile_managers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare nuScenes OSM tiles")
    parser.add_argument(
        "--nusc_dataroot",
        type=Path,
        default=Path("data/nuscenes"),
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/nuscenes/osm_tiles"),
    )
    parser.add_argument(
        "--osm_cache_dir",
        type=Path,
        default=Path("data/nuscenes/osm_cache"),
    )
    parser.add_argument("--pixel_per_meter", type=int, default=2)
    parser.add_argument("--tile_margin", type=int, default=512)
    args = parser.parse_args()

    prepare_all(
        args.nusc_dataroot,
        args.output_dir,
        version=args.version,
        osm_cache_dir=args.osm_cache_dir,
        tile_margin=args.tile_margin,
        ppm=args.pixel_per_meter,
    )
