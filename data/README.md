# Data Preparation

## Datasets

### 1. Mapillary Geo-Localization (MGL) — Training
Follow the [OrienterNet data preparation guide](https://github.com/facebookresearch/OrienterNet) to download MGL.
Place under `data/MGL/`.

### 2. KITTI — Evaluation (zero-shot)
Download KITTI odometry dataset and prepare OSM tiles following OrienterNet instructions.
Place under `data/kitti/`.

### 3. nuScenes — Evaluation (zero-shot)
Download nuScenes dataset from [nuscenes.org](https://www.nuscenes.org/).
Generate OSM tiles following OrienterNet instructions.

### 4. BDD100K — Perception Evaluation
Download BDD100K from [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/).
Expected structure:
```
bdd100k/yolop_train/
├── images/
├── bdd_det/
├── bdd_seg_gt/
└── bdd_lane_gt/
```

## OSM Tiles
All localization datasets require OpenStreetMap (OSM) raster tiles.
See OrienterNet documentation for tile generation scripts.
