"""mIoU of the graph-optimized INT8 model: (i) logits upsampled to the label as before, (ii) argmax at 1/4 res."""
import os
import sys
from pathlib import Path
sys.path.insert(0, os.environ.get("DSNET_REPO", "dsnet_src")); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import config
from data_loader import CamVidDataset, SegmentationTransform
from evaluate import evaluate_all
from ptq_eval import OrtModel
M = Path(__file__).resolve().parents[1] / "models"
ds = CamVidDataset(str(config.DATA.TEST_IMG_DIR), str(config.DATA.TEST_LABEL_DIR), transform=SegmentationTransform(config.DATA.INPUT_RESOLUTION))
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
torch.set_grad_enabled(False)
for stem in ("A_240x320_int8_percentile", "A_240x320_int8_percentile_noup_gelu"):
    m = OrtModel(M / f"{stem}.onnx", (240, 320))
    print(f"{stem:<40} (i) logits->label bilinear: mIoU {float(evaluate_all(m, loader, torch.device('cpu'))['mIoU']):.4f}", flush=True)
m = OrtModel(M / "A_240x320_int8_percentile_noup_gelu.onnx", (240, 320))
C, ig = config.DATA.NUM_CLASSES, config.DATA.IGNORE_INDEX
cm = np.zeros((C, C), np.int64)
for x, y in loader:
    p = m(x).argmax(1, keepdim=True).float()                       # decide at 60x80
    p = F.interpolate(p, size=y.shape[-2:], mode="nearest")[:, 0].long().numpy().ravel()
    t = y.numpy().ravel(); v = t != ig
    cm += np.bincount(t[v] * C + np.clip(p[v], 0, C - 1), minlength=C * C).reshape(C, C)
inter = np.diag(cm); union = cm.sum(0) + cm.sum(1) - inter
iou = [inter[c] / union[c] for c in range(C) if c != ig and union[c] > 0]
print(f"{'A_240x320_int8_percentile_noup_gelu':<40} (ii) argmax at 60x80 -> nearest: mIoU {np.mean(iou):.4f}", flush=True)
