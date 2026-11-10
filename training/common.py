"""Shared helpers: frame discovery / position groups, image preprocessing, training augmentation, metrics.

Label masks come from labels_to_masks.py: <masks_root>/<split>/<name>_mask.png (0 not-floor, 1 floor,
255 ignore), <name>_obj.png (0 none, 1..N object id), objects.json (the frame list: only reviewed frames
unless --all was used). Images: ~/mentorpi/data/ft150/<split>/<name>.jpg (640x480).
"""
import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from dsnet_model import MEAN, STD

IMG_ROOT = os.path.expanduser("~/mentorpi/data/ft150")
IGNORE = 255
LABEL_HW = (480, 640)

# Position groups. spot1 and the straight runs look at the same corridor/book -> one group (A).
GROUP_OF_SOURCE = {"spot1": "A", "straight1": "A", "straight2_rf2o": "A", "straightB": "A",
                   "spot2": "B", "spot3": "C", "spot4": "D"}


def source_of(name):
    """000_spot1_001 -> spot1, 008_straight2_rf2o_100 -> straight2_rf2o, spot4_009 -> spot4"""
    return re.sub(r"^\d{3}_", "", name).rsplit("_", 1)[0]


def group_of(name):
    src = source_of(name)
    if src not in GROUP_OF_SOURCE:
        raise KeyError(f"{name}: unknown source {src!r} (add it to GROUP_OF_SOURCE)")
    return GROUP_OF_SOURCE[src]


def labeled_frames(masks_root, splits=("train", "test")):
    """Frames listed in <masks_root>/<split>/objects.json (= reviewed frames), sorted by name within a split."""
    out = []
    for split in splits:
        oj = os.path.join(masks_root, split, "objects.json")
        if not os.path.exists(oj):
            continue
        info = json.load(open(oj, encoding="utf-8"))
        for name in sorted(info):
            out.append(frame_record(name, split, masks_root, info[name]))
    return out


def frame_record(name, split, masks_root=None, info=None):
    rec = {"name": name, "split": split, "group": group_of(name), "source": source_of(name),
           "img": os.path.join(IMG_ROOT, split, name + ".jpg"), "mask": None, "obj": None, "info": info}
    if masks_root is not None:
        rec["mask"] = os.path.join(masks_root, split, name + "_mask.png")
        rec["obj"] = os.path.join(masks_root, split, name + "_obj.png")
    return rec


def image_frames(split):
    """All frames of a split that have an image (labeled or not)."""
    d = os.path.join(IMG_ROOT, split)
    return [frame_record(os.path.splitext(f)[0], split) for f in sorted(os.listdir(d)) if f.endswith(".jpg")]


def load_rgb(path):
    im = Image.open(path).convert("RGB")
    assert im.size == (LABEL_HW[1], LABEL_HW[0]), f"{path}: unexpected size {im.size}"
    return im


def load_mask(path):
    m = np.array(Image.open(path))
    assert m.shape == LABEL_HW, f"{path}: unexpected shape {m.shape}"
    return m


def normalize(im):
    """PIL RGB -> normalized float tensor (3,H,W) (ImageNet mean/std, as in the thesis loader)."""
    x = (np.asarray(im, np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def eval_input(im, hw):
    """Evaluation / deployment preprocessing: resize the 640x480 frame to (H, W) with PIL bilinear
    (same as torchvision TF.resize on a PIL image, used in the earlier zero-shot experiments) and normalize."""
    return normalize(im.resize((hw[1], hw[0]), Image.BILINEAR))


PAD_RGB = tuple(int(round(m * 255)) for m in MEAN)  # padded image pixels normalise to ~0


class TrainSet(Dataset):
    """Augmentation: color jitter (brightness/contrast/saturation +-0.3), random scale 0.75-1.25 of the
    base 360x480 size (image bilinear, mask nearest, resized directly from the 640x480 original), random crop or
    pad back to 360x480 (image pad = mean colour, label pad = 255), horizontal flip p=0.5."""

    def __init__(self, frames, hw=(360, 480), scale=(0.75, 1.25), jitter=0.3, hflip=True):
        self.hw, self.scale, self.hflip = hw, scale, hflip
        self.imgs = [load_rgb(f["img"]) for f in frames]
        self.masks = [Image.fromarray(load_mask(f["mask"])) for f in frames]
        self.jitter = ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter) if jitter > 0 else None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img, mask = self.imgs[i], self.masks[i]
        if self.jitter is not None:
            img = self.jitter(img)
        H, W = self.hw
        s = random.uniform(*self.scale)
        h, w = int(round(H * s)), int(round(W * s))
        img = img.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)
        if w > W:
            sx, dx, cw = random.randint(0, w - W), 0, W
        else:
            sx, dx, cw = 0, random.randint(0, W - w), w
        if h > H:
            sy, dy, ch = random.randint(0, h - H), 0, H
        else:
            sy, dy, ch = 0, random.randint(0, H - h), h
        ci = Image.new("RGB", (W, H), PAD_RGB)
        ci.paste(img.crop((sx, sy, sx + cw, sy + ch)), (dx, dy))
        cm = Image.new("L", (W, H), IGNORE)
        cm.paste(mask.crop((sx, sy, sx + cw, sy + ch)), (dx, dy))
        if self.hflip and random.random() < 0.5:
            ci = ci.transpose(Image.FLIP_LEFT_RIGHT)
            cm = cm.transpose(Image.FLIP_LEFT_RIGHT)
        return normalize(ci), torch.from_numpy(np.array(cm, np.int64))


# ---------------------------------------------------------------- metrics
def binary_confusion(pred_floor, gt):
    """pred_floor: bool (H,W); gt: uint8 (H,W) with 0/1/255. Returns [[TN, FP], [FN, TP]] over valid pixels
    (rows = gt not-floor/floor, cols = pred not-floor/floor)."""
    valid = gt != IGNORE
    g = gt[valid] == 1
    p = pred_floor[valid]
    return np.array([[np.sum(~g & ~p), np.sum(~g & p)], [np.sum(g & ~p), np.sum(g & p)]], np.int64)


def scores_from_confusion(c):
    tn, fp, fn, tp = int(c[0, 0]), int(c[0, 1]), int(c[1, 0]), int(c[1, 1])
    iou_f = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    iou_n = tn / (tn + fn + fp) if tn + fn + fp else float("nan")
    tot = tn + fp + fn + tp
    return {"floor_iou": iou_f, "notfloor_iou": iou_n, "miou": (iou_f + iou_n) / 2,
            "pixel_acc": (tp + tn) / tot if tot else float("nan"),
            "false_obstacle_rate_on_floor": fn / (fn + tp) if fn + tp else float("nan"),  # true floor -> not-floor
            "false_floor_rate_on_notfloor": fp / (fp + tn) if fp + tn else float("nan"),
            "valid_px": tot}


def object_stats(pred_floor, obj, objects, thresholds=(0.5, 0.8)):
    """Per-object: fraction of the object's (visible) pixels predicted not-floor; caught at each threshold."""
    rows = []
    for k, lab in enumerate(objects or [], start=1):
        m = obj == k
        n = int(m.sum())
        if n == 0:  # fully overdrawn by a later polygon in the obj id map
            rows.append({"id": k, "label": lab, "px": 0, "notfloor_frac": None})
            continue
        rows.append({"id": k, "label": lab, "px": n, "notfloor_frac": float((~pred_floor[m]).mean())})
    obst = obj > 0
    return rows, int(obst.sum()), int((pred_floor & obst).sum())


class Evaluator:
    """Accumulates dataset-level confusion (sum over frames), per-frame scores and object statistics."""

    def __init__(self, thresholds=(0.5, 0.8)):
        self.conf = np.zeros((2, 2), np.int64)
        self.per_frame = {}
        self.objects = []
        self.obst_px = 0
        self.obst_px_pred_floor = 0
        self.thresholds = thresholds

    def add(self, name, pred_floor, gt, obj=None, objects=None):
        c = binary_confusion(pred_floor, gt)
        self.conf += c
        s = scores_from_confusion(c)
        self.per_frame[name] = {"floor_iou": s["floor_iou"], "miou": s["miou"], "pixel_acc": s["pixel_acc"]}
        if obj is not None and objects:
            rows, n_obst, n_ff = object_stats(pred_floor, obj, objects)
            for r in rows:
                r["frame"] = name
            self.objects += rows
            self.obst_px += n_obst
            self.obst_px_pred_floor += n_ff

    def summary(self):
        s = scores_from_confusion(self.conf)
        fi = [v["floor_iou"] for v in self.per_frame.values() if v["floor_iou"] == v["floor_iou"]]
        s["n_frames"] = len(self.per_frame)
        s["confusion_[[TN,FP],[FN,TP]]"] = self.conf.tolist()
        s["per_frame_floor_iou_mean"] = float(np.mean(fi)) if fi else float("nan")
        s["per_frame_floor_iou_min"] = float(np.min(fi)) if fi else float("nan")
        obj_rows = [r for r in self.objects if r["px"] > 0]
        s["n_objects"] = len(obj_rows)
        s["n_objects_hidden"] = len(self.objects) - len(obj_rows)
        for t in self.thresholds:
            s[f"object_recall@{int(t * 100)}"] = (float(np.mean([r["notfloor_frac"] >= t for r in obj_rows]))
                                                  if obj_rows else None)
        s["false_floor_rate_on_obstacle_px"] = (self.obst_px_pred_floor / self.obst_px) if self.obst_px else None
        return s

    def result(self):
        return {"summary": self.summary(), "per_frame": self.per_frame, "objects": self.objects}


def save_mask_png(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jsonable(o):
    """Make nested results JSON-safe (numpy scalars -> python, NaN -> null)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return None if o != o else o
    return o


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=1, ensure_ascii=False)


@torch.no_grad()
def predict_logits(net, frames, hw, device, bs=8):
    """Yields (frame, logits (C,480,640) float32 on CPU): input resized to hw, the model's 1/4-resolution logits
    upsampled bilinearly straight to the label resolution."""
    from dsnet_model import upsample
    net.eval()
    for i in range(0, len(frames), bs):
        chunk = frames[i:i + bs]
        x = torch.stack([eval_input(load_rgb(f["img"]), hw) for f in chunk]).to(device)
        lg = upsample(net(x).float(), LABEL_HW)
        for f, l in zip(chunk, lg):
            yield f, l.cpu()


def parse_hw(s):
    h, w = s.lower().split("x")
    return int(h), int(w)
