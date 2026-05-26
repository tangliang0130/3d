#!/usr/bin/env python3
"""Monocular depth map from a single image using Depth-Anything-V2-Small (ONNX).
Usage: python3 depth_gen.py <image> <out_depth_png> <model.onnx>
Output: grayscale depth (white = near, black = far) aligned 1:1 with the image.
"""
import sys
import numpy as np
from PIL import Image, ImageFilter
import onnxruntime as ort

img_path, out_path, model_path = sys.argv[1], sys.argv[2], sys.argv[3]

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
iname = inp.name
# Depth-Anything wants a square multiple of 14; 518 = 14*37
shape = inp.shape
S = 518
if isinstance(shape, (list, tuple)) and len(shape) == 4:
    h, w = shape[2], shape[3]
    if isinstance(h, int) and h > 0:
        S = h  # fixed-size model
print(f"input '{iname}' shape={shape} -> using {S}x{S}")

img = Image.open(img_path).convert("RGB")
W0, H0 = img.size
rimg = img.resize((S, S), Image.BILINEAR)

x = np.asarray(rimg).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
x = (x - mean) / std
x = np.transpose(x, (2, 0, 1))[None, ...]  # 1,3,S,S

out = sess.run(None, {iname: x})[0]
d = np.squeeze(out).astype(np.float32)      # S,S  (relative inverse depth: larger = closer)

# robust normalize (clip 1st/99th percentile to avoid outliers blowing contrast)
lo, hi = np.percentile(d, 1), np.percentile(d, 99)
d = np.clip((d - lo) / (hi - lo + 1e-6), 0.0, 1.0)   # 1 = near, 0 = far  (matches shader)

dimg = Image.fromarray((d * 255).astype(np.uint8), mode="L")
dimg = dimg.resize((W0, H0), Image.BILINEAR)
# light blur so displacement doesn't tear at hard depth edges
dimg = dimg.filter(ImageFilter.GaussianBlur(radius=max(1, min(W0, H0) // 350)))
dimg.save(out_path)
print(f"depth saved: {out_path}  size={dimg.size}  near=white far=black")
