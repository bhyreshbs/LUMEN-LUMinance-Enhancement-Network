# LUMEN - System architecture

```mermaid
flowchart TB
    subgraph UI["Browser - app/web/ (HTML, CSS, JS; no build, no CDN)"]
        W["Workspace /<br/>upload, style, restore, compare, download<br/>no technical terms"] --> M{Style}
        DT["How it works /details<br/>last-run details, evaluation,<br/>research lab, system status"] --> M
    end
    subgraph API["app/server.py - FastAPI"]
        H["POST /api/restore, /api/compare, /api/research<br/>upload streamed to temp file, size-limited, deleted<br/>one run at a time, queue limit -> 503<br/>CSP + security headers, JSON errors"]
    end
    M --> H
    subgraph SVC["app/service.py - EnhancementService"]
        CHK[Startup checks<br/>checkpoint load + SHA-256<br/>calibrated QGH config] --> AV[Mode availability<br/>+ reasons]
        DEC[decode_image_file<br/>lumen/imageio.py<br/>format, size, bomb, EXIF] --> RUN
        CMP[Compare All<br/>Retinexformer once<br/>QGH = refine of that output]
    end
    H -->|Natural / Highlight-safe / Quick| DEC
    H -->|Compare all| CMP
    H -->|Research| RUN
    AV -->|/api/status| M

    subgraph PIPE["lumen/pipeline.py - LumenPipeline"]
        RUN[prepare_array<br/>validate, resize in native dtype<br/>max side 768 in container] --> D{mode}
        D -->|classical| CL
        D -->|deep| DL
        D -->|quality_guided_hybrid| QG
        D -->|hybrid - research| FH
    end

    subgraph CL["lumen/classical/"]
        A1[adaptive.py<br/>image stats -> params<br/>configs/classical.yaml] --> A2[illumination.py<br/>max-RGB + guided filter<br/>Retinex I*L^(g-1)] --> A3[filters.py<br/>illumination-aware bilateral] --> A4[contrast.py<br/>CLAHE on L*, WB, saturation, roll-off]
    end

    subgraph DL["lumen/deep/"]
        R[infer.py DeepEnhancer registry<br/>pad x8, tiling, fp16 CUDA, CPU fallback<br/>CheckpointError: never random weights]
        R --> RF[retinexformer.py<br/>Retinexformer LOL-v1<br/>checkpoint mounted at /models]
        R --> ZD[zero_dce.py<br/>Zero-DCE++ - research only]
        CK[checkpoints.py<br/>known SHA-256]
    end

    subgraph QG["lumen/hybrid/quality_guided.py - QGH"]
        Q1[Retinexformer output R<br/>single model call] --> Q2[Detectors -> soft feathered masks<br/>noise, highlight enabled<br/>colour, contrast disabled by calibration]
        Q2 --> Q3{coverage >= 0.5 %?}
        Q3 -->|no| Q4[identity: return R unchanged]
        Q3 -->|yes| Q5[local corrections blended by mask<br/>guided denoise, soft-knee roll-off]
        QC[configs/quality_guided.yaml<br/>strict loader: missing/invalid -> mode disabled]
    end

    subgraph FH["lumen/hybrid/fusion.py - fusion hybrid (research baseline)"]
        F1[classical + deep + guided-deep candidates] --> F2[Mertens-style quality weights<br/>Laplacian pyramid fusion] --> F3[guided denoise + WB]
    end

    subgraph EV["Evaluation - scripts/"]
        E1[evaluate.py<br/>PSNR SSIM LPIPS NIQE latency<br/>noise, edges, colour shift, overexposure<br/>per-image CSV + QGH diagnostics]
        E2[tune_classical.py / calibrate_quality_guided.py<br/>train/dev splits only, never eval15]
        E3[benchmark_container.py / smoke_test_app.py]
    end
    PIPE --> E1
```

| Module | Responsibility |
|---|---|
| `app/app.py` | Entry point: CPU-thread and cuDNN runtime setup, service warm-up, Uvicorn |
| `app/server.py` | FastAPI: serves the two pages and static assets, JSON API (`/api/status`, `/api/examples`, `/api/restore`, `/api/compare`, `/api/research`, `/healthz`), upload streaming and deletion, single-inference gate with queue limit, security headers |
| `app/web/` | Frontend: `index.html` + `static/js/workspace.js` (workspace state machine, comparison slider, download), `details.html` + `static/js/details.js` (technical area), `static/css/lumen.css` (responsive, light/dark), `static/data/evaluation.json` (eval15 tables, mirrored from `eval_P1_T66.md` and checked by a test) |
| `app/service.py` | Mode availability (checkpoint + SHA-256, QGH config), enhancement, Compare All with one model run, Markdown summaries (unchanged by the UI redesign) |
| `lumen/imageio.py` | Safe upload decoding and array validation before any float allocation |
| `lumen/pipeline.py` | Entry point: validation, resizing, mode dispatch, model and config caching, per-run metadata |
| `lumen/classical/` | Adaptive classical pipeline (illumination, Retinex, denoise, CLAHE, colour) |
| `lumen/deep/` | Retinexformer port, Zero-DCE++, inference wrapper, checkpoint hashes |
| `lumen/hybrid/quality_guided.py` | Quality-Guided Hybrid and its calibrated configuration |
| `lumen/hybrid/fusion.py` | Fusion hybrid (research baseline) |
| `lumen/metrics/quality.py` | Full-reference, no-reference and structural metrics |
| `scripts/` | Data checks, tuning, calibration, training, evaluation, benchmarks, smoke tests, checkpoint download |
