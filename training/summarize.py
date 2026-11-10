"""Collect baselines + CV runs + main-model runs of one label snapshot into summary.json and summary.md.

usage: python summarize.py --root <runs>/<snap> [--tag _n60]
"""
import argparse
import json
import os
import re
import sys
from itertools import combinations

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dump_json, scores_from_confusion  # noqa: E402

PRE_ROOT = os.path.expanduser("~/mentorpi/data/ft150_user/prelabel_masks")


def ms(vals, nd=4):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def fmt(m, s=None, nd=4):
    if m is None:
        return "-"
    return f"{m:.{nd}f}" if s is None else f"{m:.{nd}f} ± {s:.{nd}f}"


def load_runs(root, tag):
    cv, main = {}, {}
    pat_cv = re.compile(rf"^cv_hold([A-Z])_train([A-Z]+)_s(\d+){re.escape(tag)}$")
    pat_main = re.compile(rf"^main_train([A-Z]+)_s(\d+){re.escape(tag)}$")
    for d in sorted(os.listdir(root)):
        mp = os.path.join(root, d, "metrics.json")
        if not os.path.exists(mp):
            continue
        m = pat_cv.match(d)
        if m:
            cv.setdefault(m[1], {})[int(m[3])] = json.load(open(mp)) | {"_dir": d}
            continue
        m = pat_main.match(d)
        if m:
            main[int(m[2])] = json.load(open(mp)) | {"_dir": d}
    return cv, main


def binfloor(path):
    return np.array(Image.open(path)) == 1


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    base = json.load(open(os.path.join(a.root, "baselines", "baselines.json")))
    cv, main_runs = load_runs(a.root, a.tag)
    snap = os.path.basename(os.path.normpath(a.root))
    out = {"snapshot": snap, "tag": a.tag, "labeled_per_group": {g: len(v) for g, v in base["groups"].items()},
           "folds": {}, "across_folds": {}, "pooled_cv": {}, "lowest_frames": [], "main": {}}
    hws = sorted({hw for g in cv.values() for m in g.values() for hw in m["eval"]}, key=lambda s: -int(s.split("x")[0]))
    zs_maps = list(next(iter(base["zero_shot"].values())).keys()) if base["zero_shot"] else []

    # ---------------- per fold
    for g in sorted(cv):
        runs = cv[g]
        f = {"held_out": g, "train_groups": next(iter(runs.values()))["train_groups"], "seeds": sorted(runs),
             "n_train": next(iter(runs.values()))["n_train"], "n_eval": len(next(iter(runs.values()))["eval_frames"]),
             "iters": next(iter(runs.values()))["iters"],
             "train_time_s": ms([r["train_time_s"] for r in runs.values()])[0], "finetuned": {}, "zero_shot": {},
             "eomt_prelabel": base["eomt_prelabel"]["480x640"][g]["summary"]}
        for hw in hws:
            per_seed = {s: r["eval"][hw][g]["summary"] for s, r in runs.items()}
            f["finetuned"][hw] = {k: ms([v[k] for v in per_seed.values()]) for k in
                                  ("floor_iou", "notfloor_iou", "miou", "pixel_acc", "per_frame_floor_iou_min",
                                   "false_obstacle_rate_on_floor", "false_floor_rate_on_notfloor")}
            f["finetuned"][hw]["per_seed_floor_iou"] = {s: v["floor_iou"] for s, v in per_seed.items()}
            f["finetuned"][hw]["per_seed_miou"] = {s: v["miou"] for s, v in per_seed.items()}
            if hw in base["zero_shot"]:
                f["zero_shot"][hw] = {mp: base["zero_shot"][hw][mp][g]["summary"] for mp in zs_maps}
        out["folds"][g] = f

    # ---------------- across folds (mean +- std of fold-level values) and pooled over all CV frames
    for hw in hws:
        acc = {"finetuned_floor_iou": ms([out["folds"][g]["finetuned"][hw]["floor_iou"][0] for g in cv]),
               "finetuned_miou": ms([out["folds"][g]["finetuned"][hw]["miou"][0] for g in cv]),
               "finetuned_pixel_acc": ms([out["folds"][g]["finetuned"][hw]["pixel_acc"][0] for g in cv])}
        for mp in zs_maps:
            if hw in base["zero_shot"]:
                acc[f"zero_shot_{mp}_floor_iou"] = ms([base["zero_shot"][hw][mp][g]["summary"]["floor_iou"] for g in cv])
                acc[f"zero_shot_{mp}_miou"] = ms([base["zero_shot"][hw][mp][g]["summary"]["miou"] for g in cv])
        out["across_folds"][hw] = acc
        pooled_seed = []
        seeds = sorted(set.intersection(*[set(cv[g]) for g in cv])) if cv else []
        for s in seeds:
            c = sum(np.array(cv[g][s]["eval"][hw][g]["summary"]["confusion_[[TN,FP],[FN,TP]]"]) for g in cv)
            pooled_seed.append(scores_from_confusion(c))
        pc = {"finetuned_floor_iou": ms([p["floor_iou"] for p in pooled_seed]),
              "finetuned_miou": ms([p["miou"] for p in pooled_seed])}
        if hw in base["zero_shot"]:
            for mp in zs_maps:
                c = sum(np.array(base["zero_shot"][hw][mp][g]["summary"]["confusion_[[TN,FP],[FN,TP]]"]) for g in cv)
                sc = scores_from_confusion(c)
                pc[f"zero_shot_{mp}_floor_iou"], pc[f"zero_shot_{mp}_miou"] = sc["floor_iou"], sc["miou"]
        out["pooled_cv"][hw] = pc
    out["across_folds"]["eomt_prelabel_floor_iou"] = ms([base["eomt_prelabel"]["480x640"][g]["summary"]["floor_iou"] for g in cv])
    out["across_folds"]["eomt_prelabel_miou"] = ms([base["eomt_prelabel"]["480x640"][g]["summary"]["miou"] for g in cv])
    if cv:
        c = sum(np.array(base["eomt_prelabel"]["480x640"][g]["summary"]["confusion_[[TN,FP],[FN,TP]]"]) for g in cv)
        sc = scores_from_confusion(c)
        out["pooled_cv"]["eomt_prelabel"] = {"floor_iou": sc["floor_iou"], "miou": sc["miou"]}

    # ---------------- per-frame (primary input size): lowest fine-tuned floor IoU
    if hws:
        hw = hws[0]
        rows = []
        for g in cv:
            for n in next(iter(cv[g].values()))["eval_frames"]:
                ft = [cv[g][s]["eval"][hw][g]["per_frame"][n]["floor_iou"] for s in cv[g]]
                zs = base["zero_shot"][hw]["road+sidewalk"][g]["per_frame"][n]["floor_iou"] if hw in base["zero_shot"] else None
                rows.append({"frame": n, "group": g, "finetuned_floor_iou": ms(ft)[0], "finetuned_per_seed": ft,
                             "zero_shot_rs_floor_iou": zs,
                             "eomt_floor_iou": base["eomt_prelabel"]["480x640"][g]["per_frame"][n]["floor_iou"],
                             "user_changed_frac": base["user_correction"][n]["changed_frac"]})
        rows.sort(key=lambda r: (r["finetuned_floor_iou"] is None, r["finetuned_floor_iou"]))
        out["lowest_frames"] = rows[:12]
        out["per_frame_cv"] = rows

    # ---------------- pixels the user corrected in the EoMT-L draft (held-out frames): who agrees with the user?
    plan_p = os.path.join(a.root, f"cv_plan{a.tag}.json")
    if cv and os.path.exists(plan_p):
        masks = json.load(open(plan_p))["masks"]
        zs_dir = os.path.join(a.root, "baselines", "zs_class")
        cr = {d: {"px": 0, "zs_ok": 0, "ft_ok": {}} for d in ("floor_to_notfloor", "notfloor_to_floor")}
        for g in cv:
            for n in next(iter(cv[g].values()))["eval_frames"]:
                split = "test" if n.startswith("spot4_") else "train"
                gt = np.array(Image.open(os.path.join(masks, split, n + "_mask.png")))
                pre = np.array(Image.open(os.path.join(PRE_ROOT, split, n + "_mask.png"))) == 1
                valid = gt != 255
                sel = {"floor_to_notfloor": valid & pre & (gt == 0), "notfloor_to_floor": valid & ~pre & (gt == 1)}
                if not any(m.any() for m in sel.values()):
                    continue
                zs = np.isin(np.array(Image.open(os.path.join(zs_dir, n + ".png"))), (3, 4))
                fts = {s: binfloor(os.path.join(a.root, cv[g][s]["_dir"], "pred", n + ".png")) for s in cv[g]}
                for d, m in sel.items():
                    want = d == "notfloor_to_floor"  # the user's label on these pixels: floor or not
                    cr[d]["px"] += int(m.sum())
                    cr[d]["zs_ok"] += int((zs[m] == want).sum())
                    for s, ft in fts.items():
                        cr[d]["ft_ok"][s] = cr[d]["ft_ok"].get(s, 0) + int((ft[m] == want).sum())
        out["user_corrected_pixels"] = {
            d: {"pixels": v["px"],
                "finetuned_agree_with_user": ms([k / v["px"] for k in v["ft_ok"].values()]) if v["px"] else (None, None),
                "zero_shot_rs_agree_with_user": v["zs_ok"] / v["px"] if v["px"] else None,
                "eomt_agree_with_user": 0.0} for d, v in cr.items()}

    # ---------------- main model
    if main_runs:
        r0 = next(iter(main_runs.values()))
        mm = {"train_groups": r0["train_groups"], "eval_groups": r0["eval_groups"], "seeds": sorted(main_runs),
              "n_train": r0["n_train"], "iters": r0["iters"],
              "train_time_s": ms([r["train_time_s"] for r in main_runs.values()])[0],
              "n_eval_reviewed": len(r0["eval_frames"]), "eval": {}}
        if r0["eval"]:
            for hw in r0["eval"]:
                g = r0["eval_groups"]
                summ = [r["eval"][hw][g]["summary"] for r in main_runs.values()]
                mm["eval"][hw] = {k: ms([s[k] for s in summ]) for k in
                                  ("floor_iou", "miou", "pixel_acc", "object_recall@50", "object_recall@80",
                                   "false_floor_rate_on_obstacle_px", "false_obstacle_rate_on_floor")}
                mm["eval"][hw]["n_objects"] = summ[0]["n_objects"]
                for mp in zs_maps:
                    if hw in base["zero_shot"] and g in base["zero_shot"][hw][mp]:
                        mm["eval"][hw][f"zero_shot_{mp}"] = base["zero_shot"][hw][mp][g]["summary"]
            if r0["eval_groups"] in base["eomt_prelabel"]["480x640"]:
                mm["eomt_prelabel_on_eval"] = base["eomt_prelabel"]["480x640"][r0["eval_groups"]]["summary"]
        else:
            mm["eval_note"] = r0.get("eval_note")
        # unlabeled spot4: stability across seeds and agreement with the EoMT-L draft floor (NOT ground truth)
        names = sorted(set.intersection(*[set(os.path.splitext(p)[0] for p in os.listdir(os.path.join(a.root, r["_dir"], "pred")))
                                          for r in main_runs.values()]))
        names = [n for n in names if n.startswith("spot4_")]
        preds = {s: {n: binfloor(os.path.join(a.root, r["_dir"], "pred", n + ".png")) for n in names}
                 for s, r in main_runs.items()}
        pairs = list(combinations(sorted(preds), 2))
        mm["spot4_n_predicted"] = len(names)
        mm["spot4_seed_pair_floor_iou"] = ms([iou(preds[x][n], preds[y][n]) for x, y in pairs for n in names])[0] if pairs else None
        pre = {n: binfloor(os.path.join(PRE_ROOT, "test", n + "_mask.png")) for n in names
               if os.path.exists(os.path.join(PRE_ROOT, "test", n + "_mask.png"))}
        mm["spot4_proxy_floor_iou_vs_eomt_draft"] = {
            "note": "agreement with the uncorrected EoMT-L draft (no objects drawn); a proxy, NOT a test score",
            "finetuned_per_seed": {s: ms([iou(preds[s][n], pre[n]) for n in pre])[0] for s in preds}}
        zs_dir = os.path.join(a.root, "baselines", "zs_class")
        zsf = {n: np.isin(np.array(Image.open(os.path.join(zs_dir, n + ".png"))), (3, 4)) for n in pre
               if os.path.exists(os.path.join(zs_dir, n + ".png"))}
        mm["spot4_proxy_floor_iou_vs_eomt_draft"]["zero_shot_road+sidewalk"] = ms([iou(zsf[n], pre[n]) for n in zsf])[0]
        out["main"] = mm
    dump_json(out, os.path.join(a.root, f"summary{a.tag}.json"))

    # ---------------- markdown
    L = [f"# DS-Net floor fine-tuning: leave-one-position-out CV ({snap}{a.tag})", ""]
    L.append(f"Reviewed labels per group: {out['labeled_per_group']} (D = spot4 test set). "
             f"CV folds: {sorted(cv)}; seeds per fold: {out['folds'][sorted(cv)[0]]['seeds'] if cv else '-'}. "
             "Fixed schedule, final-epoch weights, no selection on held-out data. Metrics at 480x640 "
             "(logits upsampled bilinearly), ignore=255; dataset-level IoU from summed confusion.")
    any_run = next((r for g in cv.values() for r in g.values()), None) or next(iter(main_runs.values()), None)
    if any_run:
        ar = any_run["args"]
        L.append("")
        L.append(f"Recipe: DS-Net CamVid checkpoint (all weights; epoch {any_run['init'].get('ckpt_epoch')}), decode-head "
                 f"classifier replaced by a fresh 2-class 1x1 conv; train input {ar['hw']}; aug hflip, color jitter "
                 f"±{ar['jitter']}, scale {ar['scale']} + crop/pad (label pad 255); CE ignore 255; AdamW enc "
                 f"{ar['lr_enc']:g} / head {ar['lr_head']:g}, wd {ar['wd']:g} (not on bias/norm); poly(1.0) per iter, "
                 f"no warmup; batch {ar['bs']} (drop_last); {ar['epochs']} epochs; AMP {not ar['no_amp']}.")
    L.append("")
    for hw in hws:
        L.append(f"## Held-out results, input {hw} (fine-tuned = mean ± std over seeds)")
        L.append("")
        hdr = "| held-out (train on) | n train / eval | fine-tuned floor IoU | fine-tuned mIoU |"
        sep = "|---|---|---|---|"
        for mp in zs_maps:
            hdr += f" zero-shot {mp} fIoU / mIoU |"
            sep += "---|"
        if hw == hws[0]:
            hdr += " EoMT-L pre-label fIoU / mIoU | train time / run |"
            sep += "---|---|"
        L += [hdr, sep]
        for g in sorted(cv):
            f = out["folds"][g]
            row = (f"| {g} ({f['train_groups']}) | {f['n_train']} / {f['n_eval']} | {fmt(*f['finetuned'][hw]['floor_iou'])} | "
                   f"{fmt(*f['finetuned'][hw]['miou'])} |")
            for mp in zs_maps:
                z = f["zero_shot"].get(hw, {}).get(mp)
                row += f" {fmt(z['floor_iou'])} / {fmt(z['miou'])} |" if z else " - |"
            if hw == hws[0]:
                e = f["eomt_prelabel"]
                row += f" {fmt(e['floor_iou'])} / {fmt(e['miou'])} | {f['train_time_s']:.0f} s ({f['iters']} it) |"
            L.append(row)
        acc = out["across_folds"][hw]
        row = (f"| **mean ± std over folds** | | {fmt(*acc['finetuned_floor_iou'])} | {fmt(*acc['finetuned_miou'])} |")
        for mp in zs_maps:
            if f"zero_shot_{mp}_floor_iou" in acc:
                row += f" {fmt(*acc[f'zero_shot_{mp}_floor_iou'])} / {fmt(*acc[f'zero_shot_{mp}_miou'])} |"
        if hw == hws[0]:
            row += (f" {fmt(*out['across_folds']['eomt_prelabel_floor_iou'])} / "
                    f"{fmt(*out['across_folds']['eomt_prelabel_miou'])} | |")
        L.append(row)
        pc = out["pooled_cv"][hw]
        row = f"| pooled over all CV frames | | {fmt(*pc['finetuned_floor_iou'])} | {fmt(*pc['finetuned_miou'])} |"
        for mp in zs_maps:
            if f"zero_shot_{mp}_floor_iou" in pc:
                row += f" {fmt(pc[f'zero_shot_{mp}_floor_iou'])} / {fmt(pc[f'zero_shot_{mp}_miou'])} |"
        if hw == hws[0]:
            row += f" {fmt(out['pooled_cv']['eomt_prelabel']['floor_iou'])} / {fmt(out['pooled_cv']['eomt_prelabel']['miou'])} | |"
        L += [row, ""]
    L.append("EoMT-L pre-label scores are optimistic by construction: the user's labels are corrections of these drafts.")
    L.append("")
    if out.get("user_corrected_pixels"):
        L.append("## Pixels the user corrected in the EoMT-L draft (held-out frames only)")
        L.append("")
        L.append("| correction | pixels | fine-tuned agrees with user (mean ± std over seeds) | zero-shot Road+Sidewalk | EoMT-L draft |")
        L.append("|---|---|---|---|---|")
        for d, v in out["user_corrected_pixels"].items():
            L.append(f"| {d.replace('_', ' ')} | {v['pixels']} | {fmt(*v['finetuned_agree_with_user'], nd=3)} | "
                     f"{fmt(v['zero_shot_rs_agree_with_user'], nd=3)} | 0 (by definition) |")
        L.append("")
    if out["lowest_frames"]:
        L.append(f"## Lowest held-out per-frame floor IoU (fine-tuned, {hws[0]}, mean over seeds)")
        L.append("")
        L.append("| frame | group | fine-tuned | zero-shot road+sidewalk | EoMT-L pre-label | user-changed pixels |")
        L.append("|---|---|---|---|---|---|")
        for r in out["lowest_frames"]:
            L.append(f"| {r['frame']} | {r['group']} | {fmt(r['finetuned_floor_iou'])} | {fmt(r['zero_shot_rs_floor_iou'])} | "
                     f"{fmt(r['eomt_floor_iou'])} | {r['user_changed_frac'] * 100:.2f}% |")
        L.append("")
    if out["main"]:
        mm = out["main"]
        L.append(f"## Main model: train on {mm['train_groups']} ({mm['n_train']} frames, {mm['iters']} it, "
                 f"{mm['train_time_s']:.0f} s / run), test on {mm['eval_groups']} (spot4)")
        L.append("")
        if mm["eval"]:
            for hw, e in mm["eval"].items():
                L.append(f"- [{hw}] fine-tuned floor IoU {fmt(*e['floor_iou'])}, mIoU {fmt(*e['miou'])}, "
                         f"object recall@50 {fmt(*e['object_recall@50'])}, @80 {fmt(*e['object_recall@80'])} "
                         f"({e['n_objects']} objects), false-floor on obstacle px {fmt(*e['false_floor_rate_on_obstacle_px'])}, "
                         f"false-obstacle on floor px {fmt(*e['false_obstacle_rate_on_floor'])}")
                for mp in zs_maps:
                    z = e.get(f"zero_shot_{mp}")
                    if z:
                        L.append(f"  - zero-shot {mp}: floor IoU {fmt(z['floor_iou'])}, mIoU {fmt(z['miou'])}, "
                                 f"object recall@50 {fmt(z['object_recall@50'])}, @80 {fmt(z['object_recall@80'])}")
            e = mm.get("eomt_prelabel_on_eval")
            if e:
                L.append(f"- EoMT-L pre-label on spot4: floor IoU {fmt(e['floor_iou'])}, mIoU {fmt(e['miou'])}, "
                         f"object recall@50 {fmt(e['object_recall@50'])}, @80 {fmt(e['object_recall@80'])} "
                         "(optimistic: the labels are corrections of this draft)")
        else:
            L.append(f"- spot4 metrics pending: {mm.get('eval_note')}")
        L.append(f"- predictions saved for {mm['spot4_n_predicted']} spot4 frames; mean floor IoU between seeds' "
                 f"predictions {fmt(mm['spot4_seed_pair_floor_iou'])}")
        px = mm["spot4_proxy_floor_iou_vs_eomt_draft"]
        L.append(f"- proxy only (agreement with the uncorrected EoMT-L spot4 draft, no objects drawn, NOT a test score): "
                 f"fine-tuned {', '.join(f's{s} {fmt(v)}' for s, v in px['finetuned_per_seed'].items())}; "
                 f"zero-shot road+sidewalk {fmt(px['zero_shot_road+sidewalk'])}")
        L.append("")
    with open(os.path.join(a.root, f"summary{a.tag}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
