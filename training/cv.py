"""Leave-one-position-out cross-validation + main model, run one after another on the GPU.

Fold groups = position groups with at least --min-fold-frames reviewed frames (now A, B, C; once spot4 is
labeled also D -> 4 folds, each training on the other three). Groups with fewer reviewed frames are left out
of the CV entirely and reported. Main model = every reviewed train-split frame (A+B+C), evaluated on the
reviewed spot4 frames (D) and predicting all spot4 images. Runs whose metrics.json exists are skipped (resume).

usage: python cv.py --masks <snap>/masks --out-root <runs>/<snap> [--seeds 0,1,2] [--tag _n60 -- --n-train 60]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dump_json, image_frames, labeled_frames  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def run(name, out_root, args):
    out = os.path.join(out_root, name)
    if os.path.exists(os.path.join(out, "metrics.json")):
        print(f"skip {name} (metrics.json exists)", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "ft_train.py"), "--out", out] + args
    print(f"=== {name}: {' '.join(cmd[2:])}", flush=True)
    t = time.time()
    with open(os.path.join(out, "log.txt"), "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    m = json.load(open(os.path.join(out, "metrics.json")))
    ev = m["eval"].get(next(iter(m["eval"]), ""), {})
    msg = "  ".join(f"{g}: fIoU {r['summary']['floor_iou']:.4f} mIoU {r['summary']['miou']:.4f}" for g, r in ev.items())
    print(f"    done in {time.time() - t:.0f}s (train {m['train_time_s']:.0f}s)  {msg or m.get('eval_note', '')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--min-fold-frames", type=int, default=10)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--no-main", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for run names, e.g. _n60 for a label-count curve")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="after --: extra ft_train.py arguments")
    a = ap.parse_args()
    extra = [x for x in a.extra if x != "--"]
    seeds = [int(s) for s in a.seeds.split(",") if s != ""]
    os.makedirs(a.out_root, exist_ok=True)

    frames = labeled_frames(a.masks)
    counts = Counter(f["group"] for f in frames)
    fold_groups = sorted(g for g, n in counts.items() if n >= a.min_fold_frames)
    partial = {g: n for g, n in counts.items() if n < a.min_fold_frames}
    main_train = "".join(sorted({f["group"] for f in frames if f["split"] == "train"}))
    test_groups = "".join(sorted({f["group"] for f in image_frames("test")}))
    plan = {"masks": os.path.abspath(a.masks), "labeled_per_group": dict(sorted(counts.items())),
            "fold_groups": fold_groups, "left_out_partial_groups": partial, "seeds": seeds, "extra_args": extra,
            "tag": a.tag, "runs": []}
    if partial:
        print(f"WARNING: groups with < {a.min_fold_frames} reviewed frames are left out of the CV: {partial}", flush=True)
    print(f"labeled frames per group {dict(sorted(counts.items()))}; CV folds over {fold_groups}; "
          f"main model trains on {main_train}, evaluates {test_groups}", flush=True)
    jobs = []
    if not a.no_cv and len(fold_groups) >= 2:
        for g in fold_groups:
            tr = "".join(x for x in fold_groups if x != g)
            for s in seeds:
                jobs.append((f"cv_hold{g}_train{tr}_s{s}{a.tag}",
                             ["--masks", a.masks, "--train-groups", tr, "--eval-groups", g, "--seed", str(s)] + extra))
    if not a.no_main:
        for s in seeds:
            jobs.append((f"main_train{main_train}_s{s}{a.tag}",
                         ["--masks", a.masks, "--train-groups", main_train, "--eval-groups", test_groups,
                          "--predict-split", "test", "--save-prob", "--seed", str(s)] + extra))
    plan["runs"] = [j[0] for j in jobs]
    dump_json(plan, os.path.join(a.out_root, f"cv_plan{a.tag}.json"))
    for name, args in jobs:
        run(name, a.out_root, args)
    print("all runs finished", flush=True)


if __name__ == "__main__":
    main()
