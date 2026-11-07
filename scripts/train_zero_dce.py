"""Fine-tune Zero-DCE++ with zero-reference losses on low-light images (no ground truth needed).

Example (LOL train split + LOL-v2 real train split):
    python scripts/train_zero_dce.py --data data/LOL/our485/low data/LOL-v2/Real_captured/Train/Low \
        --init weights/zero_dce_pp.pth --epochs 30

A deterministic 90/10 train/validation split is made from the given folders; the
checkpoint with the lowest validation loss is saved to weights/zero_dce_pp_finetuned.pth.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from lumen.deep.zero_dce import ZeroDCELoss, ZeroDCEpp  # noqa: E402
from lumen.utils import read_image  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class LowLightFolder(Dataset):
    def __init__(self, files: list[Path], size: int, train: bool):
        self.files, self.size, self.train = files, size, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = read_image(self.files[i])
        h, w = img.shape[:2]
        s = self.size
        if h < s or w < s:
            import cv2

            k = s / min(h, w)
            img = cv2.resize(img, (max(s, round(w * k)), max(s, round(h * k))))
            h, w = img.shape[:2]
        if self.train:
            y, x = random.randint(0, h - s), random.randint(0, w - s)
        else:
            y, x = (h - s) // 2, (w - s) // 2
        img = img[y : y + s, x : x + s]
        if self.train and random.random() < 0.5:
            img = img[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="folders of low-light images")
    ap.add_argument("--init", default=str(ROOT / "weights" / "zero_dce_pp.pth"), help="initial weights ('' = scratch)")
    ap.add_argument("--out", default=str(ROOT / "weights" / "zero_dce_pp_finetuned.pth"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--exposure", type=float, default=0.6, help="target exposure level E")
    ap.add_argument("--scale", type=int, default=1, help="Zero-DCE++ downscale factor during training")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    files = sorted(p for d in args.data for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
    if len(files) < 10:
        raise SystemExit(f"Found only {len(files)} images in {args.data}")
    random.Random(args.seed).shuffle(files)
    n_val = max(1, len(files) // 10)
    val_files, train_files = files[:n_val], files[n_val:]
    print(f"train: {len(train_files)}  val: {len(val_files)}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZeroDCEpp(scale_factor=args.scale).to(dev)
    if args.init and Path(args.init).is_file():
        model.load_state_dict(torch.load(args.init, map_location=dev, weights_only=True))
        print(f"initialised from {args.init}")
    criterion = ZeroDCELoss(exposure_level=args.exposure).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    kw = dict(batch_size=args.batch, num_workers=2, pin_memory=dev.type == "cuda")
    train_dl = DataLoader(LowLightFolder(train_files, args.size, True), shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(LowLightFolder(val_files, args.size, False), shuffle=False, **kw)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        bar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}")
        for x in bar:
            x = x.to(dev, non_blocking=True)
            enhanced, a = model(x)
            loss, parts = criterion(x, enhanced, a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()

        model.eval()
        with torch.inference_mode():
            val = [criterion(x.to(dev), *model(x.to(dev)))[0].item() for x in val_dl]
        val_loss = float(np.mean(val))
        print(f"epoch {epoch}: val_loss={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  saved best -> {args.out}")


if __name__ == "__main__":
    main()
