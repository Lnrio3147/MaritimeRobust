"""Additional synthetic corruption families, for the composition study only.

``degradations.py`` is hash-bound to every existing run and is not touched. This
module adds families the maritime protocol never used, so that composition can be
measured beyond the three hand-picked weather degradations. Each family is a
``f(image, severity, rng) -> image`` with severities 1-5 calibrated to a
monotone, roughly linear accuracy-cost ladder on the frozen baseline.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

FAMILIES = ("defocus", "motion", "noise", "rain", "darken", "contrast")


def _arr(image):
    return np.array(image.convert("RGB"), dtype=np.float32) / 255


def _img(a):
    return Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255 + 0.5))


def defocus(image, severity, rng):
    """Disc blur: radius grows with severity."""
    a = _arr(image)
    h, w = a.shape[:2]
    radius = [0.6, 1.2, 2.0, 3.0, 4.5][severity - 1] * min(h, w) / 224
    out = np.stack([gaussian_filter(a[..., c], radius) for c in range(3)], -1)
    return _img(out)


def motion(image, severity, rng):
    """Straight-line motion blur, random direction per draw."""
    a = _arr(image)
    length = [3, 6, 10, 15, 22][severity - 1]
    theta = rng.uniform(0, np.pi)
    dy, dx = int(round(np.sin(theta) * length)), int(round(np.cos(theta) * length))
    acc = np.zeros_like(a)
    count = 0
    for k in range(-max(1, abs(dx)) // 2, max(1, abs(dx)) // 2 + 1):
        for j in range(-max(1, abs(dy)) // 2, max(1, abs(dy)) // 2 + 1):
            acc += np.roll(np.roll(a, j, 0), k, 1)
            count += 1
    return _img(acc / max(1, count))


def noise(image, severity, rng):
    """Additive Gaussian sensor noise."""
    a = _arr(image)
    sigma = [0.02, 0.05, 0.09, 0.14, 0.20][severity - 1]
    return _img(a + rng.normal(0, sigma, a.shape).astype(np.float32))


def rain(image, severity, rng):
    """ImageNet-C style streaks: rotated, blurred, added."""
    a = _arr(image).copy()
    h, w = a.shape[:2]
    count = int([60, 140, 260, 420, 600][severity - 1] * h * w / (224 * 224))
    layer = np.zeros((h, w), np.float32)
    ys = rng.integers(0, h, count)
    xs = rng.integers(0, w, count)
    length = int([8, 12, 18, 25, 34][severity - 1] * h / 224) + 1
    slant = int(0.25 * length)
    for y, x in zip(ys, xs):
        layer[y:min(h, y + length), x:min(w, x + slant + 1)] = 0.85
    layer = gaussian_filter(layer, 0.8)
    alpha = [0.35, 0.5, 0.62, 0.72, 0.8][severity - 1]
    return _img(a * (1 - layer[..., None] * alpha) + layer[..., None] * alpha)


def darken(image, severity, rng):
    """Illumination loss."""
    a = _arr(image)
    gain = [0.75, 0.6, 0.45, 0.33, 0.22][severity - 1]
    return _img(a * gain)


def contrast(image, severity, rng):
    """Contrast reduction toward the image mean."""
    a = _arr(image)
    keep = [0.8, 0.62, 0.45, 0.30, 0.18][severity - 1]
    mean = a.mean(axis=(0, 1), keepdims=True)
    return _img(mean + (a - mean) * keep)


def apply(image, family, severity, seed=0):
    rng = np.random.default_rng(seed)
    return globals()[family](image, severity, rng)


def compose(image, fam_a, fam_b, severity, seed=0):
    """Both families at the same nominal severity, applied in order a then b."""
    out = apply(image, fam_a, severity, seed)
    return apply(out, fam_b, severity, seed + 1)
