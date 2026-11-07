# Model checkpoints - provenance, licences and redistribution status

Audit date: 2026-11-06. This is a record of what the upstream sources state, **not legal advice**.
"Unstated" means no licence or terms text was found in the sources listed; it does **not** mean permissive.

## Licensing matrix

| | Retinexformer `retinexformer_LOL_v1.pth` | Zero-DCE++ `zero_dce_pp.pth` |
|---|---|---|
| Used by | App (Standard, Quality Guided); evaluation | Research section only; evaluation |
| **Source-code licence** (upstream repo) | **MIT**, (c) 2023 Yuanhao Cai: `LICENSE.txt` in https://github.com/caiyuanhao1998/Retinexformer (verified via GitHub API + file) | **No licence file** in https://github.com/Li-Chongyi/Zero-DCE_extension. README: *"The code is made available for academic research purpose only. Under Attribution-NonCommercial 4.0 International License."* |
| **Checkpoint licence** | **Not explicitly stated.** The file is hosted on Google Drive (folder `1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV`), linked from the MIT repo's README. The MIT licence text covers "this software and associated documentation files"; whether it extends to the separately hosted weights is not stated | Snapshot `Epoch99.pth` is inside the repo, so the README statement (academic research only, CC BY-NC 4.0) is the only applicable text |
| **Training-dataset licence** | Trained on **LOL (v1)**, Wei et al., BMVC 2018. **Terms unstated**: no licence or terms on the project page https://daooshee.github.io/BMVC2018website/; the authors' code repo (RetinexNet) is MIT but that covers code, not the dataset. The Drive/Baidu archives themselves were not inspected | Zero-DCE++ is trained on SICE (part 1). **Not audited** |
| **Redistribution rights** | **UNCERTAIN.** Code: MIT (redistribution allowed with notice). Weights: no explicit grant; training data terms unstated. Treat as *research / academic use, with attribution*, until the authors confirm | **Not granted for this project.** CC BY-NC permits non-commercial sharing with attribution, but "academic research purpose only" is narrower. **Not redistributed by LUMEN** |
| In this repository / Docker image | **No.** Excluded by `.gitignore` and `.dockerignore`. Supplied at run time (read-only volume at `/models`, or `weights/` locally); the app verifies its SHA-256 at startup and refuses to use a missing or altered file | **No.** Excluded by `.gitignore` and `.dockerignore`; users download it themselves |
| SHA-256 | `9b4810dc51322fab70be8658ee0ad1df64a54aa814497b7849767baa2ff32cd1` | `ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84` |

Other files, which are download-only, never committed and not used at runtime:
`retinexformer_LOL_v2_real.pth` and `retinexformer_LOL_v2_synthetic.pth` (same status as the LOL-v1
checkpoint; trained on LOL-v2, whose repository https://github.com/flyywh/CVPR-2020-Semi-Low-Light states
no licence), and training checkpoints under `weights/checkpoints/` (experimental, never shipped).

## Decision (2026-11-06)

Because the checkpoint's redistribution rights are uncertain, **LUMEN does not redistribute it**: it is not
committed and not baked into the Docker image. Whoever runs LUMEN obtains it from the authors' official release
(`python scripts/download_weights.py --runtime`) and supplies it at run time:

```
docker run --rm -p 7860:7860 -v "$PWD/weights:/models:ro" lumen
```

Confirmation from the authors that the MIT licence covers the released weights would let this be revisited.

## Verify
```
python scripts/download_weights.py --runtime --verify-only   # hash + strict load, no download
```
