# Copyright (c) Li Xingyou, 2026.
# nuScenes data module for OrienterNet/MTLoc evaluation.
# Modeled after maploc/data/kitti/dataset.py

import collections.abc
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from ... import DATASETS_PATH, logger
from ...osm.tiling import TileManager
from ...utils.geo import BoundaryBox
from ..dataset import MapLocDataset
from ..torch import collate, worker_init_fn
from .prepare import MAP_ORIGINS


class NuScenesMapLocDataset(MapLocDataset):
    """MapLocDataset wrapper that ensures canvas dimensions are exact.

    Due to floating-point rounding in TileManager.query()'s round_bbox,
    the canvas can occasionally be 1 pixel off (e.g., 257x256 instead
    of 256x256). This causes a size mismatch between model scores and
    map_mask. This wrapper crops output tensors to the expected size.
    """

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        ppm = int(self.cfg.pixel_per_meter)
        expected = int(2 * self.cfg.crop_size_meters * ppm)
        m = item["map"]
        if m.shape[-2] != expected or m.shape[-1] != expected:
            item["map"] = m[:, :expected, :expected]
            if "map_mask" in item:
                item["map_mask"] = item["map_mask"][:expected, :expected]
        return item


class NuScenesDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "nuscenes",
        # paths
        "data_dir": "data/nuscenes",
        "tiles_dir": "data/nuscenes/osm_tiles",
        "version": "v1.0-mini",
        # splits
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "camera": "CAM_FRONT",
        # overwrite defaults for nuScenes
        "crop_size_meters": 64,
        "max_init_error": 30,  # DiffVL uses [-30m, +30m] uniform
        "max_init_error_rotation": 30,  # DiffVL uses [-30°, +30°]
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }

    def __init__(self, cfg, tile_managers=None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.tile_managers = tile_managers or {}
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        self.splits = {}
        self.data = {}
        self.image_paths = {}
        self.nusc = None

    def prepare_data(self):
        """Verify nuScenes data exists."""
        meta_dir = self.root / self.cfg.version
        if not meta_dir.exists():
            raise FileNotFoundError(
                f"Cannot find nuScenes metadata at {meta_dir}. "
                "Download from https://www.nuscenes.org/"
            )

    def _load_nuscenes(self):
        """Lazy-load the NuScenes object."""
        if self.nusc is None:
            from nuscenes.nuscenes import NuScenes

            self.nusc = NuScenes(
                version=self.cfg.version,
                dataroot=str(self.root),
                verbose=True,
            )
        return self.nusc

    def _get_scene_split(self):
        """Return scene tokens for train/val/test splits.

        nuScenes official splits:
        - v1.0-mini: all 10 scenes go to val (for pipeline testing)
        - v1.0-trainval: 700 train + 150 val
        """
        nusc = self._load_nuscenes()

        if self.cfg.version == "v1.0-mini":
            # Mini: use all scenes as val for pipeline testing
            all_tokens = [s["token"] for s in nusc.scene]
            return {"train": [], "val": all_tokens, "test": all_tokens}

        # For v1.0-trainval, use official splits
        try:
            from nuscenes.utils.splits import train, val

            train_names = set(train)
            val_names = set(val)
        except ImportError:
            raise ImportError("nuscenes-devkit splits not available")

        train_tokens = [
            s["token"] for s in nusc.scene if s["name"] in train_names
        ]
        val_tokens = [
            s["token"] for s in nusc.scene if s["name"] in val_names
        ]
        return {
            "train": train_tokens,
            "val": val_tokens,
            "test": val_tokens,
        }

    def setup(self, stage: Optional[str] = None):
        nusc = self._load_nuscenes()

        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]

        # Load tile managers for each location
        if not self.tile_managers:
            tiles_dir = Path(self.cfg.tiles_dir)
            for loc in MAP_ORIGINS:
                tiles_path = tiles_dir / f"tiles_{loc}.pkl"
                if tiles_path.exists():
                    logger.info("Loading tiles for %s", loc)
                    self.tile_managers[loc] = TileManager.load(tiles_path)
                else:
                    logger.warning("No tiles found for %s at %s", loc, tiles_path)

        # Get first available tile_manager to read config
        first_tm = next(iter(self.tile_managers.values()))
        self.cfg.num_classes = {k: len(g) for k, g in first_tm.groups.items()}
        self.cfg.pixel_per_meter = first_tm.ppm

        # Build scene split
        scene_splits = self._get_scene_split()

        # Pack data for each stage
        for st in stages:
            scene_tokens = scene_splits.get(st, [])
            if not scene_tokens:
                self.data[st] = {}
                self.image_paths[st] = np.array([])
                self.splits[st] = []
                continue

            self._pack_stage_data(nusc, st, scene_tokens)

    def _get_cam_front_data_for_sample(self, nusc, sample):
        """Extract CAM_FRONT sample_data, ego_pose, and calibration."""
        sd = nusc.get("sample_data", sample["data"][self.cfg.camera])
        ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        return sd, ego_pose, calib

    def _pack_stage_data(self, nusc, stage, scene_tokens):
        """Pack all sample data for a stage into tensors."""
        names = []
        t_c2w_list = []
        rpw_list = []
        camera_dicts = {}

        for scene_token in scene_tokens:
            scene = nusc.get("scene", scene_token)
            log = nusc.get("log", scene["log_token"])
            location = log["location"]
            scene_name = location  # group by location for tile_manager

            if location not in self.tile_managers:
                logger.warning(
                    "Skipping scene %s: no tiles for %s", scene["name"], location
                )
                continue

            # Initialize camera dict for this location if needed
            if scene_name not in camera_dicts:
                camera_dicts[scene_name] = {}

            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = nusc.get("sample", sample_token)
                sd, ego_pose, calib = self._get_cam_front_data_for_sample(
                    nusc, sample
                )

                # ego_pose: translation [x, y, z] in nuScenes local meters
                t_world_ego = np.array(ego_pose["translation"], dtype=np.float64)
                q_world_ego = np.array(ego_pose["rotation"], dtype=np.float64)
                # quaternion is [w, x, y, z]
                R_world_ego = Rotation.from_quat(
                    [q_world_ego[1], q_world_ego[2], q_world_ego[3], q_world_ego[0]]
                ).as_matrix()

                # Camera extrinsics (camera relative to ego vehicle)
                t_cam_ego = np.array(calib["translation"], dtype=np.float64)
                q_cam_ego = np.array(calib["rotation"], dtype=np.float64)
                R_cam_ego = Rotation.from_quat(
                    [q_cam_ego[1], q_cam_ego[2], q_cam_ego[3], q_cam_ego[0]]
                ).as_matrix()

                # Camera pose in world: T_world_cam = T_world_ego * T_ego_cam
                R_world_cam = R_world_ego @ R_cam_ego
                t_world_cam = t_world_ego + R_world_ego @ t_cam_ego

                # Extract roll, pitch, yaw from R_world_cam
                # OrienterNet convention: same as KITTI
                R_cv_xyz = Rotation.from_euler(
                    "YX", [-90, 90], degrees=True
                ).as_matrix()
                R_world_cam_xyz = R_world_cam @ R_cv_xyz
                y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler(
                    "ZYX", degrees=True
                )
                roll, pitch, yaw = r, -p, 90 - y
                roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)

                # Camera intrinsics
                K = np.array(calib["camera_intrinsic"], dtype=np.float64)
                camera_dict = {
                    "model": "PINHOLE",
                    "width": sd["width"],
                    "height": sd["height"],
                    "params": K[[0, 1, 0, 1], [0, 1, 2, 2]],
                }

                # Image path (relative to dataroot)
                image_path = sd["filename"]  # e.g. "samples/CAM_FRONT/xxx.jpg"

                # Use scene name as sequence identifier
                seq_name = scene["name"]

                # Store camera calibration per sequence
                if seq_name not in camera_dicts[scene_name]:
                    camera_dicts[scene_name][seq_name] = {}
                cam_id = 0  # single camera
                camera_dicts[scene_name][seq_name][cam_id] = camera_dict

                names.append((scene_name, seq_name, image_path))
                t_c2w_list.append(t_world_cam.astype(np.float32))
                rpw_list.append(roll_pitch_yaw)

                sample_token = sample["next"]

        if not names:
            self.data[stage] = {}
            self.image_paths[stage] = np.array([])
            self.splits[stage] = []
            return

        data = {
            "t_c2w": torch.from_numpy(np.stack(t_c2w_list)),
            "roll_pitch_yaw": torch.from_numpy(np.stack(rpw_list)),
            "camera_id": np.zeros(len(names), dtype=int),
            "cameras": camera_dicts,
        }

        self.data[stage] = data
        self.image_paths[stage] = np.array(names)
        self.splits[stage] = names

        logger.info(
            "Stage %s: %d samples from %d scenes",
            stage,
            len(names),
            len(set(n[1] for n in names)),
        )

    def dataset(self, stage: str):
        return NuScenesMapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            # image_dirs: location -> dataroot (images are loaded via relative paths)
            {loc: self.root for loc in self.tile_managers},
            self.tile_managers,
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)
