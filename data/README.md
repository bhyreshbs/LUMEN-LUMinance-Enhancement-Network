# Datasets (not committed)

Datasets listed for P1 in the hackathon datasheet. Download them manually and unpack into this folder
with the layout below (the scripts take folder paths, so other layouts also work).

| Dataset | Use | Where to get it |
|---|---|---|
| LOL (v1) | Paired eval (PSNR/SSIM/LPIPS) + fine-tuning | https://daooshee.github.io/BMVC2018website/ |
| LOL-v2 (Real + Synthetic) | Generalisation / extra training | https://github.com/flyywh/CVPR-2020-Semi-Low-Light |
| Zero-DCE implementation | Reference code + pretrained weights | https://github.com/Li-Chongyi/Zero-DCE_extension |
| MIT-Adobe FiveK | Expert-retouched references, colour fidelity | https://data.csail.mit.edu/graphics/fivek/ |

> Links were not embedded in the hackathon PDF; the URLs above are the official project pages - verify before use.

```
data/
├── LOL/
│   ├── our485/{low,high}/*.png      # train split (485 pairs)
│   └── eval15/{low,high}/*.png      # test split (15 pairs)
├── LOL-v2/
│   ├── Real_captured/{Train,Test}/{Low,Normal}/*.png
│   └── Synthetic/{Train,Test}/{Low,Normal}/*.png
├── FiveK/                           # optional
└── unseen/                          # your own / judge-like low-light photos, never used for tuning
```

**Split policy:** training/fine-tuning uses only `our485` and LOL-v2 `Train`; a 90/10 train/val split is
made automatically by `scripts/train_zero_dce.py`. `eval15`, LOL-v2 `Test` and `unseen/` are only used
for evaluation.
