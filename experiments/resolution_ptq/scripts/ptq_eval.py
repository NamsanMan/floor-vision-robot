"""PTQ accuracy for DS-Net: FP32 ONNX vs ONNX Runtime static INT8 (QDQ), CamVid test, per input size.

Same protocol as res_sweep.py: the loader yields 360x480 image/label, the image is downsized
(antialiased) to the model's fixed input size, evaluate_all upsamples logits to the label.
Quantization: QDQ, int8 activations + per-channel int8 weights (S8S8, the ARM-friendly choice),
only Conv/MatMul/Gemm (the ops that get int8 kernels); calibration on training images.

Caller sets before import: SWEEP_DATASET=camvid, SWEEP_FOLD=..., SWEEP_RESULT_BASE_DIR=<scratch>,
HF_HUB_OFFLINE=1.
"""
import argparse
import json
import random
import os
import sys
import time
from pathlib import Path

REPO = os.environ.get("DSNET_REPO", "dsnet_src")   # 논문 학습 코드(비공개) 위치
sys.path.insert(0, REPO)

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.utils.data import DataLoader

import config
from data_loader import CamVidDataset, SegmentationTransform
from eval_kd_checkpoint import load_model_checkpoint
from evaluate import evaluate_all
from models import create_model

from onnxruntime.quantization import calibrate as _calibrate


def _clear_with_merge(self):
    # ORT 1.28 bug: with CalibMaxIntermediateOutputs, collect_data() drops each chunk via
    # clear_collected_data() without computing its range first (error if calib_n % chunk == 0,
    # silent calibration on the last partial chunk otherwise). Fold the chunk into the running
    # min/max before dropping it; re-merging the final chunk is idempotent for plain min/max.
    if self.intermediate_outputs:
        self.compute_data()
    self.intermediate_outputs = []


_calibrate.MinMaxCalibrater.clear_collected_data = _clear_with_merge

METHODS = {"minmax": (CalibrationMethod.MinMax, {}),
           "percentile": (CalibrationMethod.Percentile, {"CalibPercentile": 99.999})}


def down(x, size):
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


class Logits(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, (tuple, list)) else out


class OrtModel(nn.Module):
    def __init__(self, path, size):
        super().__init__()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        self.size = size

    def forward(self, x):
        x = down(x, self.size).cpu().numpy().astype(np.float32)
        outs = [self.sess.run(None, {self.name: x[i:i + 1]})[0] for i in range(x.shape[0])]
        return torch.from_numpy(np.concatenate(outs, 0))


class Reader(CalibrationDataReader):
    def __init__(self, ds, idx, size, name):
        self.ds, self.idx, self.size, self.name, self.k = ds, list(idx), size, name, 0

    def get_next(self):
        if self.k >= len(self.idx):
            return None
        x = self.ds[self.idx[self.k]][0].unsqueeze(0)
        self.k += 1
        return {self.name: down(x, self.size).numpy().astype(np.float32)}

    def rewind(self):
        self.k = 0


def miou(model, loader):
    t0 = time.time()
    m = evaluate_all(model, loader, torch.device("cpu"))
    return float(m["mIoU"]), [None if v != v else float(v) for v in m["per_class_iou"]], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--sizes", default="360x480,240x320,180x240")
    ap.add_argument("--methods", default="minmax,percentile")
    ap.add_argument("--calib-n", type=int, default=128)
    # MinMax only: compute ranges every N samples and free them (same result, ~N/calib_n of the memory).
    ap.add_argument("--calib-chunk", type=int, default=0)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    net = create_model(config.KD.STUDENT_NAME)
    info = load_model_checkpoint(net, Path(args.ckpt), torch.device("cpu"))
    wrap = Logits(net.eval()).eval()

    tf = SegmentationTransform(config.DATA.INPUT_RESOLUTION)
    test_ds = CamVidDataset(str(config.DATA.TEST_IMG_DIR), str(config.DATA.TEST_LABEL_DIR), transform=tf)
    train_ds = CamVidDataset(str(config.DATA.TRAIN_IMG_DIR), str(config.DATA.TRAIN_LABEL_DIR), transform=tf)
    calib_idx = random.Random(0).sample(range(len(train_ds)), min(args.calib_n, len(train_ds)))
    loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f"{args.tag}: fold={config.FOLD} epoch={info.get('epoch')} test={len(test_ds)} "
          f"calib={len(calib_idx)} from train={len(train_ds)} ort={ort.__version__}", flush=True)

    res = {"tag": args.tag, "fold": config.FOLD, "ckpt": args.ckpt, "calib_n": len(calib_idx),
           "ort": ort.__version__, "quant": "QDQ S8S8 per-channel, Conv/MatMul/Gemm", "runs": []}
    for s in args.sizes.split(","):
        h, w = map(int, s.split("x"))
        size = (h, w)
        fp32 = work / f"{args.tag}_{h}x{w}_fp32.onnx"
        torch.onnx.export(wrap, torch.zeros(1, 3, h, w), str(fp32), opset_version=17,
                          input_names=["input"], output_names=["logits"], do_constant_folding=True,
                          dynamo=False)
        x0 = down(test_ds[0][0].unsqueeze(0), size)
        ref = wrap(x0).numpy()
        sess_out = OrtModel(fp32, size).sess.run(None, {"input": x0.numpy()})[0]
        diff = float(np.abs(ref - sess_out).max())
        m32, pc32, t32 = miou(OrtModel(fp32, size), loader)
        run = {"size": [h, w], "fp32_mIoU": round(m32, 4), "export_maxdiff": diff,
               "fp32_MB": round(fp32.stat().st_size / 2**20, 2), "fp32_per_class": pc32, "int8": {}}
        print(f"  {h}x{w} fp32 mIoU {m32:.4f} (export maxdiff {diff:.2e}, {t32:.0f}s)", flush=True)

        pre = work / f"{args.tag}_{h}x{w}_pre.onnx"
        try:
            quant_pre_process(str(fp32), str(pre))
        except Exception as e:  # symbolic shape inference can fail on some graphs
            print(f"    pre_process skipped: {type(e).__name__}: {e}", flush=True)
            pre = fp32
        for name in args.methods.split(","):
            method, extra = METHODS[name]
            extra = dict(extra)
            if name == "minmax" and args.calib_chunk:
                extra["CalibMaxIntermediateOutputs"] = args.calib_chunk
            q = work / f"{args.tag}_{h}x{w}_int8_{name}.onnx"
            t0 = time.time()
            quantize_static(str(pre), str(q), Reader(train_ds, calib_idx, size, "input"),
                            quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                            weight_type=QuantType.QInt8, per_channel=True,
                            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                            calibrate_method=method, extra_options=extra)
            tq = time.time() - t0
            m8, pc8, t8 = miou(OrtModel(q, size), loader)
            run["int8"][name] = {"mIoU": round(m8, 4), "delta": round(m8 - m32, 4),
                                 "MB": round(q.stat().st_size / 2**20, 2), "per_class": pc8}
            print(f"    int8 {name:<10} mIoU {m8:.4f} ({(m8 - m32) * 100:+.2f}p)  "
                  f"calib {tq:.0f}s eval {t8:.0f}s", flush=True)
        res["runs"].append(run)
        (work / f"{args.tag}_ptq.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("saved", work / f"{args.tag}_ptq.json", flush=True)


if __name__ == "__main__":
    main()
