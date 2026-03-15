# Checkpoints

Download the following checkpoints and place them in this directory:

| File | Description | Size | Link |
|------|-------------|------|------|
| `mtloc_445k.ckpt` | MTLoc best checkpoint (FPN adapter, 445K steps) | ~201 MB | [Google Drive](TODO) |
| `orienternet_mgl.ckpt` | OrienterNet pretrained on MGL (required) | ~107 MB | [OrienterNet repo](https://github.com/facebookresearch/OrienterNet) |
| `epoch-195.pth` | YOLOPX pretrained weights (required) | ~396 MB | [Google Drive](TODO) |

After downloading, your directory should look like:
```
checkpoints/
├── mtloc_445k.ckpt
├── orienternet_mgl.ckpt
└── epoch-195.pth
```
