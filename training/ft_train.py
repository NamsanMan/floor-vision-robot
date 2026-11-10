"""Fine-tune DS-Net (CamVid checkpoint) into a 2-class floor / not-floor segmenter. Plain supervised fine-tuning
(cross-entropy on the user's labels), no distillation. Fixed schedule: the FINAL-epoch weights are evaluated and
saved; nothing is selected on the evaluation frames.

usage (one fold of leave-one-position-out CV):
  python ft_train.py --masks <snap>/masks --out <run_dir> --train-groups BC --eval-groups A
main model (train on every reviewed train-split frame, evaluate reviewed spot4 frames, predict all spot4):
  python ft_train.py --masks <snap>/masks --out <run_dir> --train-groups ABC --eval-groups D --predict-split test --save-prob
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (IGNORE, Evaluator, TrainSet, dump_json, image_frames, labeled_frames, load_mask,  # noqa: E402
                    parse_hw, predict_logits, save_mask_png, seed_everything)
from dsnet_model import DSNET_CAMVID_CKPT, load_camvid_dsnet, make_two_class, save_finetuned, upsample  # noqa: E402


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="<snapshot>/masks with train/ and test/ from labels_to_masks.py")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--train-groups", required=True, help="position groups to train on, e.g. BC")
    ap.add_argument("--eval-groups", default="", help="groups to evaluate (reviewed frames only), e.g. A")
    ap.add_argument("--predict-split", default="", help="also save predictions for every image of this split")
    ap.add_argument("--n-train", type=int, default=0, help="use only the first N training frames (filename order)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr-enc", type=float, default=6e-5)
    ap.add_argument("--lr-head", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--hw", default="360x480", help="training input H x W")
    ap.add_argument("--eval-hw", default="360x480,240x320", help="input sizes to evaluate; the first is primary")
    ap.add_argument("--scale", default="0.75,1.25")
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--save-prob", action="store_true", help="also save floor probability PNGs (0-255)")
    ap.add_argument("--ckpt", default=DSNET_CAMVID_CKPT)
    return ap.parse_args(argv)


def param_groups(net, lr_enc, lr_head, wd):
    """Encoder lr / decode-head lr; weight decay on conv/linear weights only (not on biases or norm params)."""
    g = {(h, d): [] for h in (0, 1) for d in (0, 1)}
    for n, p in net.model.named_parameters():
        g[(int(n.startswith("decode_head.")), int(p.ndim > 1))].append(p)
    return [{"params": g[(0, 1)], "lr": lr_enc, "weight_decay": wd, "name": "enc"},
            {"params": g[(0, 0)], "lr": lr_enc, "weight_decay": 0.0, "name": "enc_nodecay"},
            {"params": g[(1, 1)], "lr": lr_head, "weight_decay": wd, "name": "head"},
            {"params": g[(1, 0)], "lr": lr_head, "weight_decay": 0.0, "name": "head_nodecay"}]


def evaluate(net, frames, hw, dev, groups, pred_dir=None, prob_dir=None):
    """groups: position groups to report separately (plus "all" when more than one); [] -> only "all"."""
    evs = {g: Evaluator() for g in groups}
    ev_all = Evaluator()
    for f, lg in predict_logits(net, frames, hw, dev):
        prob = torch.softmax(lg, 0)[1].numpy()
        pred = lg.argmax(0).numpy() == 1
        gt = load_mask(f["mask"])
        obj = load_mask(f["obj"])
        objs = (f["info"] or {}).get("objects")
        for ev in ([evs[f["group"]]] if f["group"] in evs else []) + [ev_all]:
            ev.add(f["name"], pred, gt, obj, objs)
        if pred_dir:
            save_mask_png(os.path.join(pred_dir, f["name"] + ".png"), pred)
        if prob_dir:
            save_mask_png(os.path.join(prob_dir, f["name"] + ".png"), np.round(prob * 255))
    out = {g: e.result() for g, e in evs.items()}
    if len(groups) != 1:
        out["all"] = ev_all.result()
    return out


def main(argv=None):
    a = get_args(argv)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    hw = parse_hw(a.hw)
    eval_hws = [parse_hw(s) for s in a.eval_hw.split(",") if s]

    frames = labeled_frames(a.masks)
    train = [f for f in frames if f["group"] in a.train_groups]
    train.sort(key=lambda f: f["name"])  # filename order: the train names are numbered so any prefix is balanced
    if a.n_train:
        if a.n_train > len(train):
            sys.exit(f"--n-train {a.n_train} > {len(train)} available training frames")
        train = train[:a.n_train]
    evalf = [f for f in frames if f["group"] in a.eval_groups] if a.eval_groups else []
    assert not set(a.train_groups) & set(a.eval_groups), "train and eval groups overlap"
    assert not {f["name"] for f in train} & {f["name"] for f in evalf}, "train and eval frames overlap"
    if not train:
        sys.exit("no training frames")
    print(f"train {len(train)} frames (groups {a.train_groups}), eval {len(evalf)} frames (groups {a.eval_groups or '-'})",
          flush=True)

    seed_everything(a.seed)
    net, info = load_camvid_dsnet(a.ckpt)
    make_two_class(net)  # fresh 2-class classifier, initialised from the seeded torch RNG
    net.to(dev)

    ds = TrainSet(train, hw=hw, scale=tuple(float(v) for v in a.scale.split(",")), jitter=a.jitter)
    gen = torch.Generator()
    gen.manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, drop_last=len(ds) >= a.bs, num_workers=a.workers,
                    persistent_workers=a.workers > 0, pin_memory=True, generator=gen)
    opt = torch.optim.AdamW(param_groups(net, a.lr_enc, a.lr_head, a.wd), betas=(0.9, 0.999), eps=1e-8)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: max(0.0, 1.0 - it / total) ** 1.0)  # poly, power 1
    amp = not a.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    log_rows = []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        tl, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = upsample(net(x), y.shape[-2:])
            loss = F.cross_entropy(logits.float(), y, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * x.size(0)
            n += x.size(0)
        row = {"epoch": ep, "loss": tl / n, "lr_enc": opt.param_groups[0]["lr"], "lr_head": opt.param_groups[2]["lr"],
               "elapsed_s": time.time() - t0}
        log_rows.append(row)
        if ep == 1 or ep % 10 == 0 or ep == a.epochs:
            print(f"ep {ep:3d}  loss {row['loss']:.4f}  lr {row['lr_enc']:.2e}/{row['lr_head']:.2e}  "
                  f"{row['elapsed_s']:.0f}s", flush=True)
    train_time = time.time() - t0
    with open(os.path.join(a.out, "train_log.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log_rows[0]))
        w.writeheader()
        w.writerows(log_rows)

    meta = {"args": vars(a), "train_frames": [f["name"] for f in train], "init": info, "epochs": a.epochs,
            "iters": total, "train_time_s": train_time}
    save_finetuned(net, os.path.join(a.out, "model_final.pth"), meta)

    t1 = time.time()
    res = {"run": os.path.basename(os.path.normpath(a.out)), "train_groups": a.train_groups,
           "eval_groups": a.eval_groups, "seed": a.seed, "n_train": len(train), "iters": total,
           "iters_per_epoch": len(dl), "train_time_s": train_time, "final_train_loss": log_rows[-1]["loss"],
           "train_frames": meta["train_frames"], "eval_frames": [f["name"] for f in evalf], "init": info,
           "args": vars(a), "eval": {}, "train_fit": {}}
    primary = f"{eval_hws[0][0]}x{eval_hws[0][1]}"
    pred_dir = os.path.join(a.out, "pred")
    prob_dir = os.path.join(a.out, "prob") if a.save_prob else None
    for d in (pred_dir, prob_dir):
        if d:
            os.makedirs(d, exist_ok=True)
    if evalf:
        egroups = sorted({f["group"] for f in evalf})
        for k, ehw in enumerate(eval_hws):
            res["eval"][f"{ehw[0]}x{ehw[1]}"] = evaluate(net, evalf, ehw, dev, egroups,
                                                         pred_dir if k == 0 else None, prob_dir if k == 0 else None)
    else:
        res["eval_note"] = f"no reviewed frames in eval groups {a.eval_groups!r} - metrics pending"
    # sanity: fit on the training frames themselves (no augmentation); NOT a generalisation estimate
    res["train_fit"][primary] = {"all": evaluate(net, train, eval_hws[0], dev, [])["all"]["summary"]}
    if a.predict_split:
        done = {f["name"] for f in evalf}
        todo = [f for f in image_frames(a.predict_split) if f["name"] not in done]
        for f, lg in predict_logits(net, todo, eval_hws[0], dev):
            save_mask_png(os.path.join(pred_dir, f["name"] + ".png"), lg.argmax(0).numpy() == 1)
            if prob_dir:
                save_mask_png(os.path.join(prob_dir, f["name"] + ".png"), np.round(torch.softmax(lg, 0)[1].numpy() * 255))
        res["predicted_unlabeled"] = [f["name"] for f in todo]
    res["eval_time_s"] = time.time() - t1
    dump_json(res, os.path.join(a.out, "metrics.json"))
    for hwk, groups in res["eval"].items():
        for g, r in groups.items():
            s = r["summary"]
            print(f"[{hwk}] eval {g}: floor IoU {s['floor_iou']:.4f}  mIoU {s['miou']:.4f}  acc {s['pixel_acc']:.4f}  "
                  f"(n={s['n_frames']})", flush=True)
    s = res["train_fit"][primary]["all"]
    print(f"train fit: floor IoU {s['floor_iou']:.4f} mIoU {s['miou']:.4f}; train {train_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
