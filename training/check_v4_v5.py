"""Check that DS-Net gives the same logits under transformers 4.x (thesis naming, ~/mentorpi/venv, CPU torch)
and 5.x (renamed keys, ~/mentorpi/venv-teacher), and that the thesis naming round-trips exactly.
Works for the CamVid checkpoint (default) and for a fine-tuned model_final.pth (3rd argument).

usage:  ~/mentorpi/venv/bin/python check_v4_v5.py dump  <out.pt> [model_final.pth]
        ~/mentorpi/venv-teacher/bin/python check_v4_v5.py compare <ref.pt> [model_final.pth]
"""
import os
import sys

import numpy as np
import torch
import transformers
from PIL import Image

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from dsnet_model import DSNET_CAMVID_CKPT, MEAN, STD, load_camvid_dsnet, load_finetuned  # noqa: E402

FRAMES = ["000_spot1_001", "001_spot2_033", "002_spot3_068", "006_straight1_102"]


def inputs():
    xs = []
    for n in FRAMES:
        im = Image.open(os.path.expanduser(f"~/mentorpi/data/ft150/train/{n}.jpg")).convert("RGB").resize((480, 360), Image.BILINEAR)
        x = (np.asarray(im, np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
        xs.append(torch.from_numpy(x.transpose(2, 0, 1)))
    return torch.stack(xs)


def main():
    mode, path = sys.argv[1], sys.argv[2]
    ft = sys.argv[3] if len(sys.argv) > 3 else None
    torch.set_num_threads(4)
    if ft:
        net, ck = load_finetuned(ft)
        saved = ck["model_state"]
        info = {"ckpt": ft, "num_classes": ck["num_classes"], "trained_with": ck.get("trained_with_transformers")}
    else:
        net, info = load_camvid_dsnet()
        saved = torch.load(DSNET_CAMVID_CKPT, map_location="cpu", weights_only=False)["model_state"]
    net.eval()
    x = inputs()
    with torch.no_grad():
        logits = net(x)
    print("transformers", transformers.__version__, "torch", torch.__version__, info, "logits", tuple(logits.shape))
    ts = net.thesis_state()
    same = set(ts) == set(saved) and all(torch.equal(ts[k], saved[k]) for k in saved)
    print("thesis-naming round trip identical to the saved model_state:", same)
    if mode == "dump":
        torch.save({"x": x, "logits": logits, "transformers": transformers.__version__}, path)
        return
    ref = torch.load(path, weights_only=False)
    assert torch.equal(ref["x"], x), "inputs differ"
    d = (ref["logits"] - logits).abs()
    agree = (ref["logits"].argmax(1) == logits.argmax(1)).float().mean().item()
    print(f"vs transformers {ref['transformers']}: max|dlogit| = {d.max().item():.3e}, mean = {d.mean().item():.3e}, "
          f"argmax agreement = {agree * 100:.4f}%")


if __name__ == "__main__":
    main()
