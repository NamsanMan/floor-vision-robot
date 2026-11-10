import json
import sys

import numpy as np
import onnxruntime as ort

model, x_path, y_path = sys.argv[1:4]
x, y = np.load(x_path), np.load(y_path)
sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
print(json.dumps({"ort_vs_pytorch_max_abs_diff": float(np.abs(out - y).max()), "shape": list(out.shape)}))
