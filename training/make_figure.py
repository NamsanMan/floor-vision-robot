"""Qualitative figures (no GPU; reads saved masks).
  fig_cv_heldout.png : per CV fold, the held-out frame with the lowest fine-tuned floor IoU and the frame the user
                       corrected most: RGB | user label | EoMT-L pre-label | DS-Net zero-shot (Road+Sidewalk) |
                       DS-Net fine-tuned (model that never saw this position) | fine-tuned errors
  fig_spot4_main.png : unlabeled spot4 test frames: RGB | EoMT-L draft | zero-shot | fine-tuned main model

usage: python make_figure.py --root <runs>/<snap> --masks <snap>/masks --prelabel <prelabel_masks_root> [--seed 0]
"""
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_ROOT  # noqa: E402

GREEN = np.array([40, 210, 90], np.float32)
MAGENTA = np.array([220, 60, 220], np.float32)
RED = np.array([235, 40, 40], np.float32)
BLUE = np.array([40, 100, 255], np.float32)


def tint(rgb, floor, ignore=None, alpha=0.45):
    out = rgb.astype(np.float32).copy()
    out[floor] = out[floor] * (1 - alpha) + GREEN * alpha
    if ignore is not None:
        out[ignore] = out[ignore] * (1 - alpha) + MAGENTA * alpha
    return out.astype(np.uint8)


def errors(rgb, pred, gt, alpha=0.6):
    out = (rgb.astype(np.float32) * 0.55).copy()
    valid = gt != 255
    ok = valid & (pred == (gt == 1)) & pred
    ff = valid & pred & (gt == 0)       # predicted floor on not-floor (dangerous: obstacle missed)
    mf = valid & ~pred & (gt == 1)      # floor missed (false obstacle)
    out[ok] = out[ok] * (1 - 0.35) + GREEN * 0.35
    out[ff] = out[ff] * (1 - alpha) + RED * alpha
    out[mf] = out[mf] * (1 - alpha) + BLUE * alpha
    return out.astype(np.uint8)


def iou(p, gt):
    v = gt != 255
    g = gt == 1
    u = ((p | g) & v).sum()
    return float(((p & g) & v).sum() / u) if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--prelabel", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-spot4", type=int, default=5)
    a = ap.parse_args()
    figdir = os.path.join(a.root, "figures")
    os.makedirs(figdir, exist_ok=True)
    base = json.load(open(os.path.join(a.root, "baselines", "baselines.json")))
    zs_dir = os.path.join(a.root, "baselines", "zs_class")

    pat = re.compile(rf"^cv_hold([A-Z])_train([A-Z]+)_s{a.seed}{re.escape(a.tag)}$")
    folds = {}
    for d in sorted(os.listdir(a.root)):
        m = pat.match(d)
        if m and os.path.exists(os.path.join(a.root, d, "metrics.json")):
            folds[m[1]] = (d, m[2], json.load(open(os.path.join(a.root, d, "metrics.json"))))
    rows = []
    for g, (d, tr, met) in sorted(folds.items()):
        pf = met["eval"][next(iter(met["eval"]))][g]["per_frame"]
        order = sorted(pf, key=lambda n: pf[n]["floor_iou"])
        corr = sorted(pf, key=lambda n: -base["user_correction"][n]["changed_px"])
        picks = [order[len(order) // 2], order[0]]  # typical (median fine-tuned floor IoU), worst
        picks.append(next(n for n in corr + order if n not in picks))  # most corrected by the user
        for n, why in zip(picks, ("median", "worst", "most user-corrected")):
            rows.append((g, tr, d, n, "train" if not n.startswith("spot4_") else "test", why))
    if rows:
        cols = ["RGB", "user label", "EoMT-L pre-label", "DS-Net zero-shot\n(Road+Sidewalk)",
                "DS-Net fine-tuned\n(held-out position)", "fine-tuned errors\nred: false floor, blue: missed floor"]
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.0 * len(cols), 2.35 * len(rows)))
        axes = np.atleast_2d(axes)
        for r, (g, tr, d, n, split, why) in enumerate(rows):
            rgb = np.array(Image.open(os.path.join(IMG_ROOT, split, n + ".jpg")).convert("RGB"))
            gt = np.array(Image.open(os.path.join(a.masks, split, n + "_mask.png")))
            pre = np.array(Image.open(os.path.join(a.prelabel, split, n + "_mask.png"))) == 1
            zs = np.isin(np.array(Image.open(os.path.join(zs_dir, n + ".png"))), (3, 4))
            ft = np.array(Image.open(os.path.join(a.root, d, "pred", n + ".png"))) == 1
            panels = [(rgb, f"{n}\nheld-out {g} (trained on {tr}), {why}"), (tint(rgb, gt == 1, gt == 255), ""),
                      (tint(rgb, pre), f"floor IoU {iou(pre, gt):.3f}"), (tint(rgb, zs), f"floor IoU {iou(zs, gt):.3f}"),
                      (tint(rgb, ft), f"floor IoU {iou(ft, gt):.3f}"), (errors(rgb, ft, gt), "")]
            for c, (img, txt) in enumerate(panels):
                ax = axes[r, c]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(cols[c], fontsize=10)
                if txt:
                    ax.set_xlabel(txt, fontsize=8.5)
        fig.suptitle("Leave-one-position-out CV: green = floor (magenta = ignore). Fine-tuned model never saw the "
                     "held-out position.", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        p = os.path.join(figdir, f"fig_cv_heldout{a.tag}.png")
        fig.savefig(p, dpi=100)
        plt.close(fig)
        print("wrote", p)

    # spot4 (labels pending): main model predictions
    mains = [d for d in sorted(os.listdir(a.root)) if re.match(rf"^main_train[A-Z]+_s{a.seed}{re.escape(a.tag)}$", d)]
    if mains:
        d = mains[0]
        names = sorted(n[:-4] for n in os.listdir(os.path.join(a.root, d, "pred")) if n.startswith("spot4_"))
        pick = [names[i] for i in np.linspace(0, len(names) - 1, min(a.n_spot4, len(names))).round().astype(int)]
        cols = ["RGB (spot4, test position)", "EoMT-L draft (no objects drawn)", "DS-Net zero-shot (Road+Sidewalk)",
                "DS-Net fine-tuned (main model)"]
        fig, axes = plt.subplots(len(pick), len(cols), figsize=(3.2 * len(cols), 2.5 * len(pick)))
        axes = np.atleast_2d(axes)
        for r, n in enumerate(pick):
            rgb = np.array(Image.open(os.path.join(IMG_ROOT, "test", n + ".jpg")).convert("RGB"))
            pre = np.array(Image.open(os.path.join(a.prelabel, "test", n + "_mask.png"))) == 1
            zs = np.isin(np.array(Image.open(os.path.join(zs_dir, n + ".png"))), (3, 4))
            ft = np.array(Image.open(os.path.join(a.root, d, "pred", n + ".png"))) == 1
            for c, img in enumerate([rgb, tint(rgb, pre), tint(rgb, zs), tint(rgb, ft)]):
                ax = axes[r, c]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(cols[c], fontsize=10)
                if c == 0:
                    ax.set_xlabel(n, fontsize=8.5)
        fig.suptitle(f"spot4 test frames (user labels pending), model {d}: green = floor", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        p = os.path.join(figdir, f"fig_spot4_main{a.tag}.png")
        fig.savefig(p, dpi=100)
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()
