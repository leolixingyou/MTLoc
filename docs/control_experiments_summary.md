# Control Experiment Results Summary

| Config | Backbone | Init | Lat@5m | Yaw@5° |
|--------|----------|------|--------|--------|
| W2 Loc-C | ELANNet | BDD100K MTL | 96.26% | 84.81% |
| Control C | ELANNet | Random | 94.12% | 80.81% |
| DenseNet | DenseNet-121 | Random | 75.75% | 47.97% |
| Control D | ResNet-50 | Random | 22.95% | 43.25% |
| Control B | ResNet-50 | ImageNet | 9.83% | 42.25% |

## Attribution
- Architecture (ELAN vs ResNet, random): 71.17pp
- Multi-path (DenseNet vs ResNet, random): 52.80pp
- ELAN-specific (ELAN vs DenseNet, random): 18.37pp
- Multi-task pretraining (MTL vs random, ELAN): 2.14pp
- ImageNet pretraining (ImageNet vs random, ResNet): -13.12pp (harmful)
