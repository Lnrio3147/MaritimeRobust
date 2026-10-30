"""The standard CIFAR-10-C corruption set, behind one callable.

``cifar_corruptions`` is the canonical implementation from
``hendrycks/robustness`` (``ImageNet-C/create_c/make_cifar_c.py``), vendored with
four compatibility patches for modern skimage/NumPy/OpenCV (gaussian's
``multichannel`` kwarg, ``np.float_``, ``cv2.resize((0, 0), ...)``, and the frost
assets shipped beside the module). Nothing about the *severity tables* or the
corruption math is changed, so a result on this set is comparable to published
CIFAR-10-C numbers.

The upstream functions disagree about their image type -- some expect a PIL
image (``motion_blur``, ``jpeg_compression``, ``pixelate`` call ``.save`` or
``.resize`` on it), the rest only read ``np.array(x)``. ``apply`` hides that:
it accepts a PIL image or a uint8 ``H x W x 3`` array and always returns a uint8
array, which is what every caller here wants.

A compound condition is ``apply`` of one family followed by another at the same
severity, matching the composition convention of the maritime grid.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from maritime_robust.external import cifar_corruptions as _c

# Canonical order, as in make_cifar_c.py. Kept in this order so a saved archive
# has the same layout as the published CIFAR-10-C tar.
NAMES = ("gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
         "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
         "brightness", "contrast", "elastic_transform", "pixelate",
         "jpeg_compression")

FUNCTIONS = {name: getattr(_c, name) for name in NAMES}

SEVERITIES = (1, 2, 3, 4, 5)


def apply(image, name, severity):
    """Apply one CIFAR-10-C corruption; returns uint8 ``H x W x 3``."""
    if not 1 <= severity <= 5:
        raise ValueError(f"severity must be 1..5, got {severity}")
    pil = image if isinstance(image, Image.Image) else Image.fromarray(
        np.asarray(image).astype(np.uint8))
    out = FUNCTIONS[name](pil, severity)
    if isinstance(out, Image.Image):
        out = np.asarray(out)
    return np.clip(np.asarray(out), 0, 255).astype(np.uint8)


def apply_pair(image, pair, severity):
    """``pair[1]`` applied on top of ``pair[0]``, both at ``severity``."""
    return apply(apply(image, pair[0], severity), pair[1], severity)


def all_pairs(names=NAMES):
    """Every unordered compound, in a stable order."""
    names = list(names)
    return [(names[i], names[j]) for i in range(len(names))
            for j in range(i + 1, len(names))]
