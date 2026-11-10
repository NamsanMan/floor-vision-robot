"""Where does INT8 change DS-Net's output? logit error, argmax flips, flip location vs class boundaries."""
import os
import sys
sys.path.insert(0, os.environ.get("DSNET_REPO", "dsnet_src"))
import numpy as np, onnx, onnxruntime as ort
from scipy.ndimage import binary_dilation
import config
from data_loader import CamVidDataset, SegmentationTransform

P = sys.argv[1]
# 1) what bit width / granularity did the quantizer actually write?
m = onnx.load(f"{P}/A_360x480_int8_percentile.onnx")
inits = {i.name: i for i in m.graph.initializer}
dt = {}
for n in m.graph.node:
    if n.op_type == "QuantizeLinear" and len(n.input) > 2 and n.input[2] in inits:
        dt.setdefault(onnx.TensorProto.DataType.Name(inits[n.input[2]].data_type), 0)
        dt[onnx.TensorProto.DataType.Name(inits[n.input[2]].data_type)] += 1
w_axes = sum(1 for n in m.graph.node if n.op_type == "DequantizeLinear" and n.input[0] in inits and len(onnx.numpy_helper.to_array(inits[n.input[1]]).shape) > 0)
print(f"zero-point dtypes of activation QuantizeLinear: {dt};  per-channel weight DequantizeLinear: {w_axes}")

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
s32 = ort.InferenceSession(f"{P}/A_360x480_fp32.onnx", so, providers=["CPUExecutionProvider"])
s8 = ort.InferenceSession(f"{P}/A_360x480_int8_percentile.onnx", so, providers=["CPUExecutionProvider"])
ds = CamVidDataset(str(config.DATA.TEST_IMG_DIR), str(config.DATA.TEST_LABEL_DIR),
                   transform=SegmentationTransform(config.DATA.INPUT_RESOLUTION))
rel, flip, near, marg_all, marg_flip = [], [], [], [], []
for i in range(0, len(ds), 7):                      # 50 images spread over the test set
    x = ds[i][0].unsqueeze(0).numpy()
    l32 = s32.run(None, {"input": x})[0][0]; l8 = s8.run(None, {"input": x})[0][0]   # (12,H,W)
    rel.append(np.linalg.norm(l8 - l32) / np.linalg.norm(l32))
    p32, p8 = l32.argmax(0), l8.argmax(0)
    f = p32 != p8
    flip.append(f.mean())
    edge = np.zeros_like(p32, bool)
    edge[:, 1:] |= p32[:, 1:] != p32[:, :-1]; edge[1:, :] |= p32[1:, :] != p32[:-1, :]
    band = binary_dilation(edge, iterations=2)       # within 2 px of a predicted class boundary
    if f.any():
        near.append(band[f].mean())
    top2 = np.sort(l32, axis=0)[-2:]
    mg = top2[1] - top2[0]
    marg_all.append(np.median(mg)); marg_flip.extend(mg[f].tolist())
    if i == 0:
        band_frac = band.mean()
print(f"images={len(rel)}")
print(f"relative logit error |l8-l32|/|l32|: mean {np.mean(rel)*100:.2f}%  max {np.max(rel)*100:.2f}%")
print(f"pixels whose predicted class flipped: mean {np.mean(flip)*100:.2f}%")
print(f"of flipped pixels, share within 2px of a class boundary: {np.mean(near)*100:.1f}%  (boundary band covers ~{band_frac*100:.0f}% of an image)")
print(f"top1-top2 logit margin: median over all pixels {np.median(marg_all):.2f}, median over flipped pixels {np.median(marg_flip):.3f}")
