# C1: Zero-Modification Baseline

No adapter training. Frozen ELANNet backbone + random FPN adapter → OrienterNet decoder.

| Metric | Value |
|--------|-------|
| Lat@5m | 0.32% |
| Yaw@5° | 34.83% |
| Lat median | 32.598m |

Key finding: Adapter training is essential for localization,
but orientation info partially exists in untrained perception features
(Yaw@5° 34.83% >> random 1.4%).
