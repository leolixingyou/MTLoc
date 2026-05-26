# YOLOPX-Loc Model Summary
## Measured from checkpoints (2026-05-26)

| Component | OrienterNet | YOLOPX-Loc | Delta |
|-----------|------------|------------|-------|
| Image Encoder | 43.26M (ResNet-101) | 13.90M (ELANNet+FPN) | -68% |
| Map Encoder | 11.72M | 11.72M | 0% |
| BEV Net | 0.07M | 0.07M | 0% |
| Total | 55.05M | 25.69M | -53% |
# Estimated FLOPs at 480x640
# ELANNet encoder: ~13 GFLOPs
# ResNet-101 encoder: ~48 GFLOPs
# Savings: -73% FLOPs, -68% params
# P1-A: OSMLoc decoder integration planned
# Target: Replace DINOv2 with ELANNet+FPN in OSMLoc framework
# Interface: model.image_encoder module replacement
# Expected: 1-2 days code + 28h training

# OSMLoc-S uses ResNet101-FPN (same encoder as OrienterNet)
# P1-A: Replace ResNet101 → ELANNet in OSMLoc-S framework
