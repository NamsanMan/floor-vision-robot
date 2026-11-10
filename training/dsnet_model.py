"""DS-Net (thesis student = SegFormer MiT-B0, ~3.7M params) model definition for fine-tuning.

Adapted from the thesis code copy ~/mentorpi/code/dsnet_src/models/segformer_wrapper.py (SegFormerWrapper,
"segformerb0" -> HuggingFace SegformerForSemanticSegmentation with the nvidia/mit-b0 config). Only the parts
needed here are kept (no KD feature hooks), and dsnet_src/config.py is NOT imported (it creates folders on
import and defaults to the KITTI profile). The CamVid class order is copied from dsnet_src/config.py
(_CAMVID_CLASS_NAMES).

The thesis checkpoint was written with transformers 4.x module names (segformer.encoder.block.i.j...,
decode_head.linear_c.i). transformers 5.x renamed the modules (segformer.stages.i.blocks.j...,
decode_head.linear_projections.i), so the state dict is renamed key by key below; shapes are unchanged.
Equivalence with transformers 4.46.3 was checked numerically (check_v4_v5.py). Fine-tuned weights are SAVED in
the thesis (4.x) naming with the "model." prefix, so they load directly into the thesis
SegFormerWrapper("segformerb0", num_classes=2).
"""
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # nvidia/mit-b0 config is in ~/.cache/huggingface

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import SegformerConfig, SegformerForSemanticSegmentation

SEGFORMER_SOURCE = "nvidia/mit-b0"
# dsnet_src/config.py: _CAMVID_CLASS_NAMES (index 11 = Void = ignore index during thesis training)
CAMVID_CLASSES = ["Sky", "Building", "Pole", "Road", "Sidewalk", "Tree", "SignSymbol", "Fence", "Car",
                  "Pedestrian", "Bicyclist", "Void"]
CAMVID_ROAD, CAMVID_SIDEWALK = 3, 4
FT_CLASSES = ["not_floor", "floor"]  # label ids of labels_to_masks.py (0 not-floor, 1 floor, 255 ignore)
MEAN = (0.485, 0.456, 0.406)  # ImageNet normalisation of the thesis loader (see main_kd._tensor_to_vis_image)
STD = (0.229, 0.224, 0.225)
DSNET_CAMVID_CKPT = os.path.expanduser("~/mentorpi/weights/dsnet/dsnet_camvid_Aset_best_model.pth")

# ---- transformers 4.x <-> 5.x key names (SegFormer) ----
_BLOCK_INNER = [  # (4.x, 5.x) prefixes inside one encoder block
    ("layer_norm_1.", "layernorm_before."),
    ("layer_norm_2.", "layernorm_after."),
    ("attention.self.query.", "attention.q_proj."),
    ("attention.self.key.", "attention.k_proj."),
    ("attention.self.value.", "attention.v_proj."),
    ("attention.self.sr.", "attention.sequence_reduction.sequence_reduction."),
    ("attention.self.layer_norm.", "attention.sequence_reduction.layer_norm."),
    ("attention.output.dense.", "attention.o_proj."),
    ("mlp.dense1.", "mlp.fc1."),
    ("mlp.dense2.", "mlp.fc2."),
    ("mlp.dwconv.dwconv.", "mlp.dwconv.dwconv."),
]


def _inner(rest, to_v5):
    for a, b in _BLOCK_INNER:
        src, dst = (a, b) if to_v5 else (b, a)
        if rest.startswith(src):
            return dst + rest[len(src):]
    raise KeyError(f"unmapped SegFormer block key part: {rest}")


def v4_to_v5_key(k):
    m = re.fullmatch(r"segformer\.encoder\.patch_embeddings\.(\d+)\.(.+)", k)
    if m:
        return f"segformer.stages.{m[1]}.patch_embeddings.{m[2]}"
    m = re.fullmatch(r"segformer\.encoder\.layer_norm\.(\d+)\.(.+)", k)
    if m:
        return f"segformer.stages.{m[1]}.layer_norm.{m[2]}"
    m = re.fullmatch(r"segformer\.encoder\.block\.(\d+)\.(\d+)\.(.+)", k)
    if m:
        return f"segformer.stages.{m[1]}.blocks.{m[2]}.{_inner(m[3], True)}"
    m = re.fullmatch(r"decode_head\.linear_c\.(\d+)\.(.+)", k)
    if m:
        return f"decode_head.linear_projections.{m[1]}.{m[2]}"
    return k


def v5_to_v4_key(k):
    m = re.fullmatch(r"segformer\.stages\.(\d+)\.patch_embeddings\.(.+)", k)
    if m:
        return f"segformer.encoder.patch_embeddings.{m[1]}.{m[2]}"
    m = re.fullmatch(r"segformer\.stages\.(\d+)\.layer_norm\.(.+)", k)
    if m:
        return f"segformer.encoder.layer_norm.{m[1]}.{m[2]}"
    m = re.fullmatch(r"segformer\.stages\.(\d+)\.blocks\.(\d+)\.(.+)", k)
    if m:
        return f"segformer.encoder.block.{m[1]}.{m[2]}.{_inner(m[3], False)}"
    m = re.fullmatch(r"decode_head\.linear_projections\.(\d+)\.(.+)", k)
    if m:
        return f"decode_head.linear_c.{m[1]}.{m[2]}"
    return k


def adapt_state_dict(sd, target_keys):
    """Rename a SegformerForSemanticSegmentation state dict (no "model." prefix) to the naming of target_keys."""
    target_keys = set(target_keys)
    if set(sd) == target_keys:
        return dict(sd)
    for fn in (v4_to_v5_key, v5_to_v4_key):
        out = {fn(k): v for k, v in sd.items()}
        if set(out) == target_keys:
            return out
    raise RuntimeError("state dict keys do not match this transformers version even after renaming: "
                       f"missing {sorted(target_keys - set(sd))[:5]} unexpected {sorted(set(sd) - target_keys)[:5]}")


class DSNet(nn.Module):
    """SegformerForSemanticSegmentation(nvidia/mit-b0 config): same architecture as the thesis SegFormerWrapper.
    forward(x) returns logits at 1/4 input resolution (callers upsample bilinearly, like the wrapper does)."""

    def __init__(self, num_classes):
        super().__init__()
        cfg = SegformerConfig.from_pretrained(SEGFORMER_SOURCE)
        names = CAMVID_CLASSES if num_classes == len(CAMVID_CLASSES) else (
            FT_CLASSES if num_classes == 2 else [f"class_{i}" for i in range(num_classes)])
        cfg.id2label = {i: n for i, n in enumerate(names)}
        cfg.label2id = {n: i for i, n in enumerate(names)}
        cfg.num_labels = num_classes
        self.model = SegformerForSemanticSegmentation(cfg)  # weights come from the DS-Net checkpoint, not HF
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(pixel_values=x).logits

    def load_thesis_state(self, model_state):
        """model_state: thesis SegFormerWrapper.state_dict() (keys "model.<hf key>", transformers 4.x names)."""
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in model_state.items()}
        sd = adapt_state_dict(sd, self.model.state_dict().keys())
        self.model.load_state_dict(sd, strict=True)

    def thesis_state(self):
        """State dict in the thesis naming (prefix "model.", transformers 4.x module names)."""
        return {"model." + v5_to_v4_key(k): v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


def load_camvid_dsnet(path=DSNET_CAMVID_CKPT):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = DSNet(len(CAMVID_CLASSES))
    net.load_thesis_state(ck["model_state"])
    info = {"ckpt": path, "ckpt_epoch": ck.get("epoch"), "ckpt_best_test_mIoU": ck.get("best_test_mIoU"),
            "n_params": sum(p.numel() for p in net.parameters())}
    del ck
    return net, info


def make_two_class(net):
    """Replace the decode-head classifier (1x1 Conv2d 256->12) with a fresh 2-class layer; keep all other weights.
    Uses the global torch RNG (seeded by the caller)."""
    old = net.model.decode_head.classifier
    net.model.decode_head.classifier = nn.Conv2d(old.in_channels, 2, kernel_size=1)
    net.model.config.num_labels = 2
    net.model.config.id2label = {0: FT_CLASSES[0], 1: FT_CLASSES[1]}
    net.model.config.label2id = {FT_CLASSES[0]: 0, FT_CLASSES[1]: 1}
    net.num_classes = 2
    return net


def save_finetuned(net, path, meta):
    torch.save({"model_state": net.thesis_state(), "num_classes": net.num_classes, "classes": FT_CLASSES,
                "key_format": "thesis SegFormerWrapper(segformerb0) state_dict, transformers 4.x module names",
                "trained_with_transformers": transformers.__version__, "mean": MEAN, "std": STD, **meta}, path)


def load_finetuned(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = DSNet(ck.get("num_classes", 2))
    net.load_thesis_state(ck["model_state"])
    return net, ck


def upsample(logits, hw):
    return F.interpolate(logits, size=hw, mode="bilinear", align_corners=False)
