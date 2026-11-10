"""미세조정 DS-Net의 Pi 배포 모델 만들기 (서브컴 venv: ORT 1.23.2 = 로봇과 같은 버전). 10/07 실험(2026-10-07_dsnet_resolution_ptq)과 같은 방식:
1) ONNX FP32 (입력 1x3xHxW, 출력 H/4xW/4 로짓 = 마지막 업샘플 없음, 10/07의 '_noup'과 같음), opset 17
2) INT8 정적 양자화: QDQ, 가중치 채널별 int8 + 활성값 int8(S8S8), Conv/MatMul/Gemm만(LayerNorm·Softmax·GELU는 FP32),
   Percentile 99.999, 보정 = train 사진 N장(시드 0). 메모리: 240x320은 장당 약 120MB → 48장 약 6GB(서브컴 15GB)
3) GELU 묶기(onnxruntime.transformers FusionGelu): QDQ 그래프에선 ORT가 GELU를 자동으로 못 묶음(10/07 발견, Pi -18%)
4) spot4 test 정확도: torch FP32 / ORT FP32 / ORT INT8 / ORT INT8+GELU (바닥 IoU, 접지 띠 recall, 주행 가능 거리 오차)
usage: ~/mentorpi/venv/bin/python deploy_int8.py <model_final.pth> <masks_root> <out_dir> [--hw 240x320] [--calib-n 48]"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.transformers.fusion_gelu import FusionGelu
from onnxruntime.transformers.onnx_model import OnnxModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_ROOT, LABEL_HW, eval_input, load_rgb  # noqa: E402
from contact_metrics import evaluate, load_gt  # noqa: E402
from dsnet_model import FT_CLASSES, load_finetuned  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("masks_root")
ap.add_argument("out")
ap.add_argument("--hw", default="240x320")
ap.add_argument("--calib-n", type=int, default=48)
a = ap.parse_args()
H, W = map(int, a.hw.split("x"))
os.makedirs(a.out, exist_ok=True)
torch.set_grad_enabled(False)
FLOOR = FT_CLASSES.index("floor")
stem = os.path.join(a.out, f"ft_{H}x{W}")
log = {"ckpt": a.ckpt, "hw": [H, W], "ort": ort.__version__, "torch": torch.__version__,
       "quant": "QDQ S8S8 per-channel, Conv/MatMul/Gemm, Percentile 99.999", "calib_n": a.calib_n}

net = load_finetuned(a.ckpt)[0].eval()
fp32 = stem + "_fp32.onnx"
kw = dict(opset_version=17, input_names=["input"], output_names=["logits"], do_constant_folding=True)
try:
    torch.onnx.export(net, torch.zeros(1, 3, H, W), fp32, dynamo=False, **kw)
except TypeError:
    torch.onnx.export(net, torch.zeros(1, 3, H, W), fp32, **kw)

train_imgs = sorted(glob.glob(os.path.join(IMG_ROOT, "train", "*.jpg")))
calib = random.Random(0).sample(train_imgs, min(a.calib_n, len(train_imgs)))


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = iter(calib)

    def get_next(self):
        p = next(self.it, None)
        return None if p is None else {"input": eval_input(load_rgb(p), (H, W)).unsqueeze(0).numpy()}


pre = stem + "_pre.onnx"
try:
    quant_pre_process(fp32, pre)
except Exception as e:
    print("pre_process skipped:", type(e).__name__, e)
    pre = fp32
int8 = stem + "_int8.onnx"
t0 = time.time()
quantize_static(pre, int8, Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=True, op_types_to_quantize=["Conv", "MatMul", "Gemm"], calibrate_method=CalibrationMethod.Percentile,
                extra_options={"CalibPercentile": 99.999})
log["calib_sec"] = round(time.time() - t0, 1)
om = OnnxModel(onnx.load(int8))
FusionGelu(om).apply()
om.prune_graph()
if not any(o.domain == "com.microsoft" for o in om.model.opset_import):
    om.model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
int8g = stem + "_int8_gelu.onnx"
onnx.save(om.model, int8g)
cnt = lambda m, op: sum(1 for n in m.graph.node if n.op_type == op)  # noqa: E731
log["nodes"] = {"int8_QuantizeLinear": cnt(onnx.load(int8), "QuantizeLinear"), "int8_Erf": cnt(onnx.load(int8), "Erf"),
                "int8_gelu_Gelu": cnt(om.model, "Gelu"), "int8_gelu_Erf": cnt(om.model, "Erf")}
log["MB"] = {k: round(os.path.getsize(p) / 2**20, 2) for k, p in (("fp32", fp32), ("int8_gelu", int8g))}

names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(IMG_ROOT, "test", "*.jpg")))
gt, obj, objn = load_gt(a.masks_root, names)
X = {n: eval_input(load_rgb(os.path.join(IMG_ROOT, "test", n + ".jpg")), (H, W)).unsqueeze(0) for n in names}


def to_floor(logits):
    lg = F.interpolate(torch.as_tensor(logits).float(), size=LABEL_HW, mode="bilinear", align_corners=False)[0]
    return lg.argmax(0).numpy() == FLOOR


res = {"torch_fp32": evaluate({n: to_floor(net(X[n])) for n in names}, gt, obj, objn, names)}
ref = {n: net(X[n]).numpy() for n in names}
so = ort.SessionOptions()
so.intra_op_num_threads = 4
for tag, p in (("ort_fp32", fp32), ("ort_int8", int8), ("ort_int8_gelu", int8g)):
    s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
    out = {n: s.run(None, {"input": X[n].numpy()})[0] for n in names}
    res[tag] = evaluate({n: to_floor(out[n]) for n in names}, gt, obj, objn, names)
    res[tag]["max_abs_logit_diff_vs_torch"] = round(float(max(np.abs(out[n] - ref[n]).max() for n in names)), 4)
    res[tag]["pixel_agree_vs_torch"] = round(float(np.mean([(to_floor(out[n]) == to_floor(ref[n])).mean() for n in names])), 5)
log["results"] = res
json.dump(log, open(stem + "_deploy.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(log, indent=1, ensure_ascii=False))
