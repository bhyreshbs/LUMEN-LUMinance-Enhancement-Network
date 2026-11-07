"""Supervised fine-tuning of Retinexformer on paired low/normal-light data.

    python scripts/train_retinexformer.py                          # LOL-v1 our485 + any LOL-v2 Train present
    python scripts/train_retinexformer.py --sets lol_v1 lol_v2_real --epochs 30
    python scripts/train_retinexformer.py --resume                 # continue from weights/checkpoints/

Data
  * Training pool: LOL-v1 `our485` plus LOL-v2 Real/Synthetic `Train` folders that exist under data/.
  * Any training pair whose low or high image duplicates a *test* image (LOL-v1 eval15, LOL-v2 Test)
    is dropped (16x16 thumbnail fingerprint), so test images can never leak into training.
  * Exact duplicates inside the pool are removed (LOL-v2-real Train re-contains ~400 LOL-v1 images), and a
    deterministic validation split (`--val-frac`, `--seed`) is drawn by near-duplicate scene group.

Training
  * Random 256 px crops, then a random 128-256 px (multiple of 32) sub-crop per optimiser step,
    random flips + 90-degree rotations.
  * loss = L1 + 0.2 * (1 - SSIM) + 0.05 * LPIPS(alex)
  * AdamW, linear warm-up + cosine decay, fp16 AMP with GradScaler, gradient accumulation (4 GB VRAM).

Selection
  * The starting (official) weights are scored on the validation split first. The fine-tuned weights are
    written to weights/retinexformer_finetuned.pth *only if* they beat that validation PSNR, and
    DeepEnhancer("retinexformer") picks that file up automatically.
  * Every epoch a full resumable checkpoint goes to weights/checkpoints/retinexformer_last.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from lumen.deep.retinexformer import build_lol_model, load_checkpoint  # noqa: E402

DATA = ROOT / "data"
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
TRAIN_SETS = {
    "lol_v1": ("LOL/our485/low", "LOL/our485/high"),
    "lol_v2_real": ("LOL-v2/Real_captured/Train/Low", "LOL-v2/Real_captured/Train/Normal"),
    "lol_v2_syn": ("LOL-v2/Synthetic/Train/Low", "LOL-v2/Synthetic/Train/Normal"),
}
TEST_SETS = [
    ("LOL/eval15/low", "LOL/eval15/high"),
    ("LOL-v2/Real_captured/Test/Low", "LOL-v2/Real_captured/Test/Normal"),
    ("LOL-v2/Synthetic/Test/Low", "LOL-v2/Synthetic/Test/Normal"),
]
CKPT_DIR = ROOT / "weights" / "checkpoints"
FINETUNED = ROOT / "weights" / "retinexformer_finetuned.pth"


# ----------------------------------------------------------------------------- data
def read_u8(path) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"unreadable image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pair_dir(low: Path, high: Path) -> list[tuple[Path, Path]]:
    """Pair by identical name, else by the digits in the name (LOL-v2: low00690 <-> normal00690)."""
    if not (low.is_dir() and high.is_dir()):
        return []
    highs = [p for p in sorted(high.iterdir()) if p.suffix.lower() in EXTS]
    by_name, by_num = {p.name: p for p in highs}, {re.sub(r"\D", "", p.stem): p for p in highs}
    out = []
    for lo in sorted(low.iterdir()):
        if lo.suffix.lower() in EXTS:
            hi = by_name.get(lo.name) or by_num.get(re.sub(r"\D", "", lo.stem))
            if hi is not None:
                out.append((lo, hi))
    return out


def fingerprint(path) -> np.ndarray:
    g = cv2.cvtColor(read_u8(path), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).ravel()


def drop_test_duplicates(pairs, tol: float = 0.01):
    test = [fingerprint(p) for lo, hi in TEST_SETS for pair in pair_dir(DATA / lo, DATA / hi) for p in pair]
    if not test:
        return pairs, 0
    test = np.stack(test)
    kept, dropped = [], 0
    for lo, hi in pairs:
        f = np.stack([fingerprint(lo), fingerprint(hi)])
        d = np.abs(f[:, None, :] - test[None]).mean(-1)  # 2 x n_test mean abs difference
        if (d < tol).any():
            dropped += 1
        else:
            kept.append((lo, hi))
    return kept, dropped


def group_split(pairs, val_frac: float, seed: int, exact_tol: float = 0.002, group_tol: float = 0.01):
    """Drop exact duplicates inside the pool, then split by near-duplicate *groups*.

    LOL-v2-real Train contains ~400 copies of LOL-v1 our485 and neighbouring shots of the same scene,
    so a plain random split would put copies of an image on both sides of the train/val boundary.
    Union-find over 16x16 thumbnails of the normal-light images keeps each scene in one split.
    """
    fps = np.stack([fingerprint(hi) for _, hi in pairs])
    keep = []
    for i in range(len(pairs)):  # exact duplicates: keep the first occurrence
        if not keep or np.abs(fps[keep] - fps[i]).mean(-1).min() >= exact_tol:
            keep.append(i)
    n_exact = len(pairs) - len(keep)
    pairs, fps = [pairs[i] for i in keep], fps[keep]

    parent = list(range(len(pairs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(pairs)):
        close = np.nonzero(np.abs(fps[i + 1:] - fps[i]).mean(-1) < group_tol)[0] + i + 1
        for j in close:
            parent[find(int(j))] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(pairs)):
        groups.setdefault(find(i), []).append(i)
    order = sorted(groups.values(), key=lambda g: g[0])
    random.Random(seed).shuffle(order)
    target = round(len(pairs) * val_frac)
    val_idx, train_idx = [], []
    for g in order:
        (val_idx if len(val_idx) < target else train_idx).extend(g)
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx], n_exact, len(groups)


class PairedCrops(Dataset):
    def __init__(self, pairs, crop: int = 256):
        self.pairs, self.crop = pairs, crop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        lo, hi = read_u8(self.pairs[i][0]), read_u8(self.pairs[i][1])
        if lo.shape != hi.shape:
            hi = cv2.resize(hi, (lo.shape[1], lo.shape[0]), interpolation=cv2.INTER_AREA)
        h, w = lo.shape[:2]
        c = self.crop
        if min(h, w) < c:  # upscale tiny images so a full crop fits
            k = c / min(h, w)
            size = (math.ceil(w * k), math.ceil(h * k))
            lo, hi = cv2.resize(lo, size, interpolation=cv2.INTER_CUBIC), cv2.resize(hi, size, interpolation=cv2.INTER_CUBIC)
            h, w = lo.shape[:2]
        y, x = random.randint(0, h - c), random.randint(0, w - c)
        lo, hi = lo[y:y + c, x:x + c], hi[y:y + c, x:x + c]
        k = random.randint(0, 3)  # dihedral augmentation: rotations + flips
        lo, hi = np.rot90(lo, k), np.rot90(hi, k)
        if random.random() < 0.5:
            lo, hi = lo[:, ::-1], hi[:, ::-1]
        if random.random() < 0.5:
            lo, hi = lo[::-1], hi[::-1]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))  # noqa: E731
        return t(lo), t(hi)


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


# ----------------------------------------------------------------------------- losses
def _gauss(ws=11, sigma=1.5, device="cpu"):
    g = torch.exp(-((torch.arange(ws, device=device) - ws // 2) ** 2) / (2 * sigma**2))
    g = (g / g.sum())[:, None]
    return (g @ g.T)[None, None].repeat(3, 1, 1, 1)


def ssim_torch(x, y, window):
    c1, c2 = 0.01**2, 0.03**2
    p = window.shape[-1] // 2
    mu_x, mu_y = F.conv2d(x, window, padding=p, groups=3), F.conv2d(y, window, padding=p, groups=3)
    sxx = F.conv2d(x * x, window, padding=p, groups=3) - mu_x**2
    syy = F.conv2d(y * y, window, padding=p, groups=3) - mu_y**2
    sxy = F.conv2d(x * y, window, padding=p, groups=3) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return s.mean()


class RestorationLoss(torch.nn.Module):
    def __init__(self, w_ssim=0.2, w_lpips=0.05):
        super().__init__()
        import lpips

        self.w_ssim, self.w_lpips = w_ssim, w_lpips
        self.lpips = lpips.LPIPS(net="alex", verbose=False).eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        self.register_buffer("window", _gauss())

    def forward(self, pred, gt):
        pred = pred.float()  # losses in fp32 for stability
        l1 = F.l1_loss(pred, gt)
        ssim = ssim_torch(pred.clamp(0, 1), gt, self.window)
        lp = self.lpips(pred.clamp(0, 1) * 2 - 1, gt * 2 - 1).mean()
        total = l1 + self.w_ssim * (1 - ssim) + self.w_lpips * lp
        return total, {"l1": l1.item(), "ssim": ssim.item(), "lpips": lp.item()}


# ----------------------------------------------------------------------------- validation
@torch.inference_mode()
def validate(model, pairs, device, fp16: bool) -> float:
    model.eval()
    scores = []
    for lo, hi in pairs:
        x = torch.from_numpy(read_u8(lo).transpose(2, 0, 1).copy())[None].float().div(255).to(device)
        y = torch.from_numpy(read_u8(hi).transpose(2, 0, 1).copy())[None].float().div(255).to(device)
        h, w = x.shape[-2:]
        xp = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode="reflect")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            out = model(xp)[..., :h, :w].float().clamp(0, 1)
        if y.shape != out.shape:
            y = F.interpolate(y, size=out.shape[-2:], mode="area")
        mse = F.mse_loss(out, y).item()
        scores.append(10 * math.log10(1.0 / max(mse, 1e-10)))
    model.train()
    return float(np.mean(scores))


# ----------------------------------------------------------------------------- main
def lr_lambda(step, warmup, total, min_ratio):
    if step < warmup:
        return (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=list(TRAIN_SETS), choices=list(TRAIN_SETS))
    ap.add_argument("--init", default=str(ROOT / "weights" / "retinexformer_LOL_v1.pth"),
                    help="starting weights ('' = train from scratch)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=400, help="optimiser steps per epoch")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation (effective batch = batch*accum)")
    ap.add_argument("--crop-min", type=int, default=128)
    ap.add_argument("--crop-max", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5, help="peak LR (official from-scratch LR is 2e-4)")
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=200, help="warm-up optimiser steps")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--w-ssim", type=float, default=0.2)
    ap.add_argument("--w-lpips", type=float, default=0.05)
    ap.add_argument("--dedup-tol", type=float, default=0.01,
                    help="drop train pairs whose 16x16 thumbnail is within this mean diff of a test image (0 = off)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", action="store_true", help=f"resume from {CKPT_DIR / 'retinexformer_last.pt'}")
    ap.add_argument("--max-train-pairs", type=int, default=0, help="debug: limit the training pool")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and not args.no_amp

    # ---- data
    pool = []
    for name in args.sets:
        lo, hi = TRAIN_SETS[name]
        found = pair_dir(DATA / lo, DATA / hi)
        print(f"{name:12s}: {len(found)} pairs" + ("" if found else "  (not found - skipped)"))
        pool += found
    pool, dropped = drop_test_duplicates(pool, args.dedup_tol) if args.dedup_tol > 0 else (pool, 0)
    print(f"dropped {dropped} training pairs that (near-)duplicate test images")
    if len(pool) < 20:
        raise SystemExit("Not enough training pairs.")
    train_pairs, val_pairs, n_exact, n_groups = group_split(pool, args.val_frac, args.seed)  # deterministic
    print(f"removed {n_exact} exact duplicates inside the pool; {n_groups} scene groups split train/val")
    if args.max_train_pairs:
        train_pairs = train_pairs[: args.max_train_pairs]
    val_real = [v for v in val_pairs if "Synthetic" not in str(v[0])]
    val_syn = [v for v in val_pairs if "Synthetic" in str(v[0])]
    print(f"train: {len(train_pairs)}  val: {len(val_pairs)} (real {len(val_real)}, synthetic {len(val_syn)})  "
          f"device: {dev}  amp: {amp}")

    def val_breakdown(m):
        r = validate(m, val_real, dev, amp) if val_real else float("nan")
        y = validate(m, val_syn, dev, amp) if val_syn else float("nan")
        n_r, n_y = len(val_real), len(val_syn)
        overall = (np.nan_to_num(r) * n_r + np.nan_to_num(y) * n_y) / max(1, n_r + n_y)
        return float(overall), r, y

    loader = DataLoader(
        PairedCrops(train_pairs, args.crop_max), batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=dev.type == "cuda",
        worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(args.seed),
    )

    # ---- model / optimiser
    model = build_lol_model().to(dev)
    if args.init and Path(args.init).is_file():
        load_checkpoint(model, args.init, map_location=dev)
        print(f"initialised from {args.init}")
    criterion = RestorationLoss(args.w_ssim, args.w_lpips).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    total_steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup, total_steps, args.min_lr_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    last_path = CKPT_DIR / "retinexformer_last.pt"
    log_path = ROOT / "results" / "training" / "retinexformer_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch, step = 1, 0

    if args.resume:
        if not last_path.is_file():
            raise SystemExit(f"No checkpoint to resume at {last_path}")
        ck = torch.load(last_path, map_location=dev, weights_only=False)  # our own file: contains RNG states
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        best_val, init_val = ck["best_val"], ck["init_val"]
        random.setstate(ck["py_rng"])
        np.random.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"])
        print(f"resumed at epoch {start_epoch}, step {step}; init val {init_val:.3f}, best val {best_val:.3f}")
    else:
        if FINETUNED.exists():  # a previous run's result must not silently stay the default
            prev = CKPT_DIR / "retinexformer_finetuned_prev.pth"
            FINETUNED.replace(prev)
            print(f"moved previous {FINETUNED.name} -> {prev}")
        t0 = time.perf_counter()
        init_val, r0, y0 = val_breakdown(model)
        best_val = init_val
        print(f"starting weights - val PSNR {init_val:.3f} dB (real {r0:.3f}, synthetic {y0:.3f}) "
              f"({time.perf_counter() - t0:.1f}s)")
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "lr", "loss", "l1", "ssim", "lpips", "val_psnr",
                                    "val_real", "val_syn", "sec"])
            csv.writer(f).writerow([0, 0, 0, "", "", "", "", round(init_val, 4), round(r0, 4), round(y0, 4), 0])

    model.train()
    it = iter(loader)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        agg = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "lpips": 0.0}
        for _ in range(args.steps_per_epoch):
            s = random.randrange(args.crop_min, args.crop_max + 1, 32)  # one patch size per step
            opt.zero_grad(set_to_none=True)
            for _ in range(args.accum):
                try:
                    lo, hi = next(it)
                except StopIteration:
                    it = iter(loader)
                    lo, hi = next(it)
                y0 = random.randint(0, args.crop_max - s)
                x0 = random.randint(0, args.crop_max - s)
                lo = lo[..., y0:y0 + s, x0:x0 + s].to(dev, non_blocking=True).float().div_(255)
                hi = hi[..., y0:y0 + s, x0:x0 + s].to(dev, non_blocking=True).float().div_(255)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    pred = model(lo)
                loss, parts = criterion(pred, hi)
                scaler.scale(loss / args.accum).backward()
                agg["loss"] += loss.item() / args.accum
                for k in ("l1", "ssim", "lpips"):
                    agg[k] += parts[k] / args.accum
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

        n = args.steps_per_epoch
        val, vr, vy = val_breakdown(model)
        sec = time.perf_counter() - t0
        lr = sched.get_last_lr()[0]
        print(f"epoch {epoch:3d}/{args.epochs}  loss {agg['loss'] / n:.4f}  ssim {agg['ssim'] / n:.4f}  "
              f"lpips {agg['lpips'] / n:.4f}  lr {lr:.2e}  val PSNR {val:.3f} [real {vr:.2f} syn {vy:.2f}] (start {init_val:.3f}, "
              f"best {max(best_val, val):.3f})  {sec:.0f}s", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, step, f"{lr:.3e}", *(round(agg[k] / n, 5) for k in agg), round(val, 4),
                                    round(vr, 4), round(vy, 4), round(sec, 1)])

        if val > best_val:
            best_val = val
            torch.save({"params": model.state_dict(), "val_psnr": val, "init_val_psnr": init_val,
                        "epoch": epoch, "args": vars(args)}, FINETUNED)
            print(f"  new best (+{val - init_val:.3f} dB over starting weights) -> {FINETUNED}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "step": step, "best_val": best_val,
                    "init_val": init_val, "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(), "args": vars(args)}, last_path)

    summary = {"init_val_psnr": init_val, "best_val_psnr": best_val, "improved": best_val > init_val,
               "finetuned_written": best_val > init_val, "train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
               "dropped_test_duplicates": dropped, "pool_exact_duplicates": n_exact,
               "scene_groups": n_groups}
    (log_path.parent / "retinexformer_summary.json").write_text(json.dumps(summary, indent=2))
    if best_val > init_val:
        print(f"\nFine-tuning improved val PSNR {init_val:.3f} -> {best_val:.3f} dB; saved {FINETUNED}")
    else:
        print(f"\nFine-tuning did not beat the starting weights on val ({init_val:.3f} dB); nothing written.")


if __name__ == "__main__":
    main()
