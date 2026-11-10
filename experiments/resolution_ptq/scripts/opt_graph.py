"""Graph-level speedups on existing ONNX models (no retraining, no re-quantization).

noup : drop the final Resize (logits 1/4 res -> input res). Output becomes 1/4-res logits.
gelu : fuse Div-Erf-Add-Mul-Mul into com.microsoft.Gelu. ORT fuses this itself for the FP32
       graph but not for the QDQ INT8 graph, where the 8 GELUs stay unfused (Erf ~15% of time).

usage: python opt_graph.py <models_dir> <stem> [<stem> ...]   e.g. A_240x320_int8_percentile
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnxruntime.transformers.fusion_gelu import FusionGelu
from onnxruntime.transformers.onnx_model import OnnxModel


def drop_final_resize(model):
    g = model.graph
    out = g.output[0]
    resize = next(n for n in g.node if out.name in n.output)
    assert resize.op_type == "Resize", resize.op_type
    src = resize.input[0]
    om = OnnxModel(model)
    om.remove_node(resize)
    out.name = src
    out.type.tensor_type.ClearField("shape")  # 1/4-res shape; let ORT infer it
    om.prune_graph()
    return om.model


def fuse_gelu(model):
    om = OnnxModel(model)
    FusionGelu(om).apply()
    om.prune_graph()
    if not any(o.domain == "com.microsoft" for o in om.model.opset_import):
        om.model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    return om.model


def run(path, x):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x})[0]


def count(model, op):
    return sum(1 for n in model.graph.node if n.op_type == op)


def main():
    d = Path(sys.argv[1])
    for stem in sys.argv[2:]:
        src = d / f"{stem}.onnx"
        base = onnx.load(str(src))
        h, w = [x.dim_value for x in base.graph.input[0].type.tensor_type.shape.dim[2:]]
        x = np.random.default_rng(0).standard_normal((1, 3, h, w)).astype(np.float32)
        ref = run(src, x)
        variants = {"noup": drop_final_resize(onnx.load(str(src))),
                    "gelu": fuse_gelu(onnx.load(str(src)))}
        variants["noup_gelu"] = fuse_gelu(drop_final_resize(onnx.load(str(src))))
        for tag, m in variants.items():
            p = d / f"{stem}_{tag}.onnx"
            onnx.save(m, str(p))
            y = run(p, x)
            if y.shape[-2:] != ref.shape[-2:]:  # compare after the same bilinear upsample the dropped Resize did
                y = F.interpolate(torch.from_numpy(y), size=ref.shape[-2:], mode="bilinear", align_corners=False).numpy()
            print(f"{p.name:<44} out {tuple(run(p, x).shape)}  Gelu {count(m, 'Gelu')} Erf {count(m, 'Erf')} "
                  f"Resize {count(m, 'Resize')}  max|diff| vs original {np.abs(y - ref).max():.2e}")


if __name__ == "__main__":
    main()
