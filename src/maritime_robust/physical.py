"""Physically-motivated re-rendering of haze and sun glare.

A validity check on the approximations in ``degradations.py``: nothing here is
imported by the training pipeline, and no existing run is re-scored. Two pieces
of physics are added that the original renderer omits.

*Haze* — Koschmieder scattering with transmission that falls with distance,
``t = exp(-beta * d(y))``, where ``d`` is a ground-plane depth prior estimated
per image from the horizon row. The original renderer uses a nearly uniform
transmission and a random near-white airlight; here the airlight is the median
colour of the sky region.

*Glare* — a specular sun lobe on the water plus a sensor response: the saturated
mask blooms through a Gaussian and is fed back before the final clip. The
original renderer adds a Gaussian gain and clips, with no blooming.

Severity is matched on measurable quantities rather than on hand-set constants:
haze by mean transmission, glare by saturated-pixel fraction, both to the value
the original renderer produces on the same images. The comparison is therefore
"same pixel statistics, different physical structure".

Usage is through ``render``; see reports/PREREG_PHYSICAL_RENDERER.md.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter1d

SATURATED = 254 / 255


def horizon_row(a):
    """Row of strongest horizontal edge in the upper 80%, as a depth anchor.

    Maritime framing puts the sky/sea boundary there; the estimate is a
    heuristic and P3 of the pre-registration measures its consequences.
    """
    grey = a.mean(2)
    energy = uniform_filter1d(np.abs(np.diff(grey, axis=0)).mean(1), 3)
    lo, hi = int(0.10 * len(energy)), int(0.80 * len(energy))
    return lo + int(np.argmax(energy[lo:hi]))


def depth_field(h, w, horizon):
    """Ground-plane depth proxy: 1 at the bottom of frame, 1 for sky."""
    rows = np.clip((np.arange(h)[:, None] - horizon) / max(1, h - 1 - horizon), 0, 1)
    depth = np.repeat(rows.astype(np.float32), w, axis=1)
    depth[np.arange(h) <= horizon] = 1.0
    return depth


def airlight_of(a, horizon):
    """Median colour of the sky region, floored so it stays visible."""
    sky = a[:max(2, horizon)]
    if sky.size == 0:
        return np.array([0.75, 0.78, 0.82], np.float32)
    return np.clip(np.median(sky.reshape(-1, 3), axis=0), 0.45, 0.98).astype(np.float32)


def beta_for_target(depth, target):
    """Solve exp(-beta * depth).mean() == target by bisection."""
    lo, hi = 1e-4, 60.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if np.exp(-mid * depth).mean() > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def apply_haze(a, target_transmission, depth=None):
    horizon = horizon_row(a)
    airlight = airlight_of(a, horizon)
    if depth is None:
        depth = depth_field(*a.shape[:2], horizon)
    beta = beta_for_target(depth, target_transmission)
    t = np.exp(-beta * depth)[..., None]
    return a * t + airlight * (1.0 - t), {"horizon_row": int(horizon),
                                          "airlight": airlight.tolist(),
                                          "beta": float(beta),
                                          "transmission_mean": float(t.mean())}


def apply_glare(a, target_saturation, rng, bloom_gain=0.35, iterations=14):
    """Specular sun lobe on the water, then a blooming sensor response.

    The exposure gain is solved per image so the saturated fraction matches
    ``target_saturation``; ``bloom_gain`` spills the clipped mask into its
    neighbours the way a real sensor does.
    """
    h, w = a.shape[:2]
    horizon = horizon_row(a)
    cx = float(rng.uniform(0.25, 0.75))
    # the specular image of the sun sits below the horizon by the depression angle
    cy = horizon + (h - horizon) * float(rng.uniform(0.15, 0.55))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lobe = np.exp(-(((xx - cx * w) / (0.30 * w)) ** 2
                    + ((yy - cy) / (0.14 * h)) ** 2) / 2)
    streak = np.exp(-((xx - cx * w) / (0.035 * w)) ** 2 / 2) * np.exp(
        -((yy - cy) / (0.55 * h)) ** 2 / 2)
    shape = 0.65 * lobe + 0.35 * streak
    tint = np.array([1.0, 0.98, 0.92], np.float32)

    def render(gain):
        lit = a + (gain * shape)[..., None] * tint
        hard = (lit.max(2) >= SATURATED).astype(np.float32)
        bloom = gaussian_filter(hard, max(1.0, 0.04 * min(h, w)))
        out = np.clip(lit + (bloom_gain * bloom)[..., None], 0, 1)
        # the quantity being matched is the saturated fraction of the *delivered*
        # image, blooming included, exactly as the original renderer reports it
        return out, (out.max(2) >= SATURATED).astype(np.float32)

    lo, hi = 0.0, 20.0
    gain = 0.0
    out, mask = render(gain)
    if target_saturation > 0:
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            out, mask = render(mid)
            if mask.mean() < target_saturation:
                lo = mid
            else:
                hi = mid
        gain = 0.5 * (lo + hi)
        out, mask = render(gain)
    return out, {"glint_center": [cx, float(cy / h)], "gain": float(gain),
                 "saturated_fraction": float(mask.mean())}


def render(image, kind, target_transmission, target_saturation, seed=0,
           uniform_haze=False):
    """Re-render one image. ``kind`` is haze, glare or haze+glare."""
    a = np.array(image.convert("RGB"), dtype=np.float32) / 255
    rng = np.random.default_rng(seed)
    horizon = horizon_row(a)
    depth = depth_field(*a.shape[:2], horizon)
    params = {}
    for part in kind.split("+"):
        if part == "haze":
            use_depth = None if not uniform_haze else np.ones_like(depth)
            a, info = apply_haze(a, target_transmission, use_depth)
            params.update({"haze_" + k: v for k, v in info.items()})
        elif part == "glare":
            a, info = apply_glare(a, target_saturation, rng)
            params.update({"glare_" + k: v for k, v in info.items()})
        else:
            raise ValueError(part)
    params["saturated_fraction"] = float((a.max(2) >= SATURATED).mean())
    return Image.fromarray(np.uint8(a * 255 + 0.5)), params
