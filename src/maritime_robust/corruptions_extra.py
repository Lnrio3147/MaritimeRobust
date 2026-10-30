"""Additional synthetic corruption families, for the composition study only.

``degradations.py`` is hash-bound to every existing run and is not touched. This
module adds families the maritime protocol never used, so that composition can be
measured beyond the three hand-picked weather degradations.

Every family takes a continuous ``level`` in ``[0, 1]`` rather than a discrete
severity label, because severities are not comparable across families: at the
same nominal level these corruptions cost very different accuracy. The study
equates families on the *measured* marginal drop (see
``scripts/composition_calibrate.py``) and only then pairs them.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

FAMILIES = ("defocus", "motion", "noise", "rain", "pixelate", "impulse")

# ``darken`` and ``contrast`` are kept for the record but excluded from the
# study: a global gain leaves ConvNeXt's LayerNorm-normalised tokens unchanged,
# so both plateau near zero cost until they are extreme (<= 20 pp at level 1.0,
# against 30-60 pp for every other family at the same level).


def _arr(image):
    return np.array(image.convert("RGB"), dtype=np.float32) / 255


def _img(a):
    return Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255 + 0.5))


def defocus(image, level, rng):
    """Disc blur: radius grows linearly with the level."""
    a = _arr(image)
    h, w = a.shape[:2]
    radius = max(0.05, 9.0 * level) * min(h, w) / 224
    return _img(np.stack([gaussian_filter(a[..., c], radius) for c in range(3)], -1))


def motion(image, level, rng):
    """Straight-line motion blur, random direction per draw."""
    a = _arr(image)
    length = max(1, int(round(44 * level * min(a.shape[:2]) / 224)))
    theta = rng.uniform(0, np.pi)
    dy, dx = int(round(np.sin(theta) * length)), int(round(np.cos(theta) * length))
    acc = np.zeros_like(a)
    count = 0
    for k in range(-max(0, dx // 2), max(0, dx // 2) + 1):
        for j in range(-max(0, dy // 2), max(0, dy // 2) + 1):
            acc += np.roll(np.roll(a, j, 0), k, 1)
            count += 1
    return _img(acc / max(1, count))


def noise(image, level, rng):
    """Additive Gaussian sensor noise."""
    a = _arr(image)
    return _img(a + rng.normal(0, 0.35 * level, a.shape).astype(np.float32))


def rain(image, level, rng):
    """Streak layer, ImageNet-C style, blurred and composited.

    Density is fixed and the compositing weight carries the level, so the
    damage is monotone: letting the streak count grow turns heavy rain into a
    near-white field, which the classifier finds *easier* than moderate rain.
    """
    a = _arr(image).copy()
    h, w = a.shape[:2]
    count = int(400 * h * w / (224 * 224))
    layer = np.zeros((h, w), np.float32)
    ys = rng.integers(0, h, count)
    xs = rng.integers(0, w, count)
    length = int(25 * h / 224) + 1
    slant = int(0.25 * length)
    for y, x in zip(ys, xs):
        layer[y:min(h, y + length), x:min(w, x + slant + 1)] = 0.9
    layer = gaussian_filter(layer, 0.8)
    alpha = 0.10 + 0.85 * level
    return _img(a * (1 - layer[..., None] * alpha) + layer[..., None] * alpha)


def darken(image, level, rng):
    """Illumination loss."""
    return _img(_arr(image) * (1.0 - 0.97 * level))


def contrast(image, level, rng):
    """Contrast reduction toward the image mean."""
    a = _arr(image)
    mean = a.mean(axis=(0, 1), keepdims=True)
    return _img(mean + (a - mean) * (1.0 - 0.97 * level))


def pixelate(image, level, rng):
    """Spatial quantisation: detail is destroyed, global statistics are kept."""
    a = _arr(image)
    h, w = a.shape[:2]
    k = max(2, int(round((1.0 - 0.94 * level) * min(h, w))))
    small = Image.fromarray(np.uint8(a * 255 + 0.5)).resize((k, k), Image.Resampling.NEAREST)
    return small.resize((w, h), Image.Resampling.NEAREST)


def impulse(image, level, rng):
    """Salt-and-pepper: a fraction of pixels is driven to the extremes."""
    a = _arr(image).copy()
    rate = 0.30 * level
    mask = rng.random(a.shape[:2])
    a[mask < rate / 2] = 0.0
    a[mask > 1 - rate / 2] = 1.0
    return _img(a)


def apply(image, family, level, seed=0):
    rng = np.random.default_rng(seed)
    return globals()[family](image, level, rng)


def compose(image, fam_a, fam_b, level, seed=0):
    """Both families at the same level, applied in order a then b."""
    out = apply(image, fam_a, level, seed)
    return apply(out, fam_b, level, seed + 1)
