# Third-party notices

Audit date 2026-11-06; not legal advice. Full checkpoint provenance and the redistribution status are in
[`weights/README.md`](weights/README.md).

| Component | What LUMEN contains | Source-code licence | Data / weights licence | Redistributed by LUMEN |
|---|---|---|---|---|
| Retinexformer architecture | `lumen/deep/retinexformer.py` (port of `RetinexFormer_arch.py`) | MIT (text below) | - | Yes (code, with this notice) |
| Retinexformer LOL-v1 checkpoint | not included; supplied by the operator at run time (`/models` volume) | - | **Not explicitly stated**; trained on LOL, whose terms are unstated. Redistribution rights **uncertain** | **No** |
| Zero-DCE++ architecture | `lumen/deep/zero_dce.py` (re-implementation using the original layer names so the official snapshot loads) | Upstream: academic research only, CC BY-NC 4.0 (README statement; no licence file) | - | Yes (re-implemented code). **Keep its use academic and non-commercial.** See note |
| Zero-DCE++ checkpoint | not included | - | academic research only, CC BY-NC 4.0 | **No** |
| LOL / LOL-v2 datasets | not included (`data/` is ignored); `data/splits/*.txt` lists file names only | - | terms unstated | No |
| Example images | `app/examples/*.jpg`, synthetic low-light derivatives | - | CC0 / public domain sources (`app/examples/SOURCES.md`) | Yes |
| Hackathon problem statement | not included (organisers' PDF and the P1 excerpt are git-ignored) | - | organisers' document; no licence stated | **No** |

**Repository status:** private. LUMEN's own code has **no licence file** by deliberate choice, so no reuse
rights are granted.

## Python dependencies (licence as declared in package metadata, versions in requirements*.txt)

| Package | Licence | Where used |
|---|---|---|
| numpy | BSD-3-Clause (+0BSD, MIT, Zlib, CC0 parts) | runtime |
| opencv-contrib-python-headless | Apache-2.0 | runtime |
| scikit-image, scipy | BSD | runtime |
| PyWavelets | MIT AND BSD-3-Clause | runtime |
| Pillow | MIT-CMU | runtime |
| PyYAML | MIT | runtime |
| torch | Apache-2.0 / BSD (mixed) | runtime |
| fastapi | MIT | runtime (web API) |
| starlette, uvicorn | BSD-3-Clause | runtime (web server) |
| python-multipart | Apache-2.0 | runtime (upload parsing) |
| pydantic, pydantic-core, anyio, h11, annotated-types, typing-inspection, annotated-doc | MIT | runtime (FastAPI/Uvicorn dependencies) |
| click, idna | BSD-3-Clause | runtime (FastAPI/Uvicorn dependencies) |
| typing-extensions | PSF-2.0 | runtime (dependency) |
| httpx | BSD-3-Clause | tests only (FastAPI TestClient) |
| **pyiqa** | **PolyForm-Noncommercial-1.0.0** | optional NIQE metric (`requirements-niqe.txt`); **not in the default image**. It also downloads NIQE model parameters at first use |
| torchvision | BSD | NIQE extra, evaluation |
| lpips | BSD (declared "UNKNOWN" in metadata, BSD classifier) | evaluation only. Downloads AlexNet weights via torchvision |
| optuna, gdown, pytest | MIT | development only |
| tqdm | MPL-2.0 AND MIT | development only |
| psutil | BSD-3-Clause | development only |

## Must be resolved or replaced before any public or commercial release

| # | Item | Issue | Options |
|---|---|---|---|
| 1 | `lumen/deep/zero_dce.py` (network + zero-reference losses) and `scripts/train_zero_dce.py` | Re-implements Zero-DCE++ with the upstream layer names; upstream terms are "academic research purpose only", CC BY-NC 4.0 | Obtain permission; or remove Zero-DCE++ (it is research-only and not used by the default modes); or write a clean-room implementation from the paper with different parameter names (then the official snapshot can no longer be loaded) |
| 2 | Retinexformer checkpoint | Weights have no explicit licence; training data (LOL) terms unstated | Not redistributed: deployers supply it. For public hosting, ask the authors to confirm the MIT licence covers the released weights |
| 3 | pyiqa (NIQE) | PolyForm Noncommercial | Keep NIQE optional and non-commercial; or replace it with a permissively licensed NIQE implementation |
| 4 | LUMEN's own code | No licence file | Choose a licence before publishing, compatible with items 1 and 3 |
| 5 | `configs/quality_guided.yaml`, `configs/classical.yaml`, `data/splits/dev_lolv2_real_unseen.txt` | Derived from LOL / LOL-v2 (calibrated thresholds, tuned constants, file names); dataset terms unstated | Low risk (no images or labels); mention the datasets with citation, as done here |

Code reviewed as **not** requiring action: `lumen/deep/retinexformer.py` (MIT port, notice retained); guided
filter, LIME-style illumination, multi-scale Retinex, CLAHE usage, Mertens-style fusion and quality-guided
refinement are LUMEN's own implementations of published algorithms (no third-party code copied); example
images are CC0 / public domain.

**Note on `lumen/deep/zero_dce.py`:** it reproduces the published network structure and the upstream
parameter names, which makes it arguably a derivative of the upstream code. Until LUMEN has its own licence
that addresses this, treat the file as subject to the upstream academic, non-commercial terms. LUMEN's own
code currently has **no LICENSE file**, so by default no reuse rights are granted. Choose a licence before
publishing; MIT would need an exception for `zero_dce.py`.

## Retinexformer - MIT licence (verbatim)
Source: https://github.com/caiyuanhao1998/Retinexformer/blob/master/LICENSE.txt

```
MIT License

Copyright (c) 2023 Yuanhao Cai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Zero-DCE++ - upstream statement (verbatim, README of https://github.com/Li-Chongyi/Zero-DCE_extension)
> The code is made available for academic research purpose only. Under Attribution-NonCommercial 4.0 International License.

Citation: C. Li, C. Guo, C. C. Loy, "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve
Estimation", TPAMI 2021.

## Dataset citations
- LOL: C. Wei, W. Wang, W. Yang, J. Liu, "Deep Retinex Decomposition for Low-Light Enhancement", BMVC 2018.
- LOL-v2: W. Yang, H. Huang, W. Wang, S. Wang, J. Liu, "Sparse Gradient Regularized Deep Retinex Network for
  Robust Low-Light Image Enhancement", TIP 2021.
