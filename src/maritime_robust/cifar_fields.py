"""Continuous corruption fields: the representation behind docs/MODULE_NCF.md.

A degradation is written as ``x_deg = clamp(K * x_clean + N)`` with a spatial
gain ``K`` in [0, 1] and an additive field ``N``. The point of the
representation is that families are *regions* of a continuous space rather than
enumeration items, so a prior over (K, N) can in principle cover a family it was
never told about. Whether it actually can is the question the capacity probe in
``scripts/cifar_field_probe.py`` answers, before any of the module is built.

Nothing here is a method yet: this file holds the prior, the compositor and the
fitter, all of which are measured by the probe.
"""
from __future__ import annotations

import numpy as np


def smooth_field(rng: np.random.Generator, h: int = 32, w: int = 32,
                 octaves: int = 3) -> np.ndarray:
    """A low-frequency random field in [0, 1], upsampled from coarse noise."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        size = max(2, (2 ** (octaves - o)) + 1)
        coarse = rng.random((size, size)).astype(np.float32)
        ys = np.linspace(0, size - 1, h)
        xs = np.linspace(0, size - 1, w)
        y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
        y1, x1 = np.minimum(y0 + 1, size - 1), np.minimum(x0 + 1, size - 1)
        wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
        top = coarse[y0][:, x0] * (1 - wx) + coarse[y0][:, x1] * wx
        bot = coarse[y1][:, x0] * (1 - wx) + coarse[y1][:, x1] * wx
        out += (top * (1 - wy) + bot * wy) * amp
        total += amp
        amp *= 0.5
    out /= total
    return np.clip(out, 0, 1)


def sample_blur(rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """Per-image convolution, as three numbers: (sigma_x, sigma_y, theta).

    Blur is a convolution, not a pixel-wise gain, so no (K, N) on any grid can
    express it. Measured on the trained estimator, every blur family lands at or
    *below* AugMix (motion 83.2 vs 92.6, glass 75.7 vs 88.4) while every
    photometric and noise family reaches the supervised level -- the boundary is
    exactly "does this degradation convolve". One small, global kernel per image
    removes that boundary without naming a family: isotropic sigma reads as
    defocus, a long thin kernel reads as motion, both are drawn from
    distributions, not from a list.
    """
    if rng.random() >= p:
        return np.zeros(3, np.float32)
    if rng.random() < 0.5:                       # isotropic: defocus-like
        s = rng.uniform(0.45, 2.4)
        return np.array([s, s, 0.0], np.float32)
    sx, sy = rng.uniform(1.2, 3.2), rng.uniform(0.35, 0.9)   # motion-like
    return np.array([sx, sy, rng.uniform(0.0, np.pi)], np.float32)


def gaussian_kernel(params: np.ndarray, size: int = 11) -> np.ndarray:
    """Normalised 2-D Gaussian from (sigma_x, sigma_y, theta); identity if zero."""
    sx, sy, th = (float(v) for v in params)
    if sx <= 1e-3 and sy <= 1e-3:
        k = np.zeros((size, size), np.float32)
        k[size // 2, size // 2] = 1.0
        return k
    sx, sy = max(sx, 1e-3), max(sy, 1e-3)
    r = size // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    xr = xx * np.cos(th) + yy * np.sin(th)
    yr = -xx * np.sin(th) + yy * np.cos(th)
    k = np.exp(-(xr ** 2 / (2 * sx ** 2) + yr ** 2 / (2 * sy ** 2)))
    return (k / k.sum()).astype(np.float32)


def apply_blur(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Convolve an HxWx3 image in [0, 1] with the kernel; reflect padding."""
    sx, sy = float(params[0]), float(params[1])
    if sx <= 1e-3 and sy <= 1e-3:
        return x
    k = gaussian_kernel(params)
    r = k.shape[0] // 2
    pad = np.pad(x, ((r, r), (r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(x)
    for dy in range(k.shape[0]):
        for dx in range(k.shape[1]):
            weight = k[dy, dx]
            if weight == 0.0:
                continue
            out += weight * pad[dy:dy + x.shape[0], dx:dx + x.shape[1]]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def sample_field(rng: np.random.Generator, h: int = 32, w: int = 32,
                 transmission: bool = True, glare: bool = True,
                 noise: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Draw (K, N, sigma) from the continuous prior of docs/MODULE_NCF.md §3.2.

    Every component is parameterised physically (transmission level, ambient
    light, blob geometry, noise scale). No component is named after a CIFAR-10-C
    family, which is what makes the held-out-family test in §5 meaningful.
    """
    K = np.ones((h, w), np.float32)
    N = np.zeros((h, w, 3), np.float32)

    # Some deployment images are undegraded. Without this branch the estimator
    # has never seen K = 1, N = 0 and reports a small spurious field on a clean
    # image (measured: mean K 0.98, |N| 0.019), which is pure added error when
    # that field is applied to a different picture.
    if rng.random() < 0.12:
        return K, N, 0.0

    if transmission and rng.random() < 0.7:
        level = rng.uniform(0.15, 1.0)
        structure = smooth_field(rng, h, w, int(rng.integers(2, 5)))
        K *= 1.0 - (1.0 - level) * structure
        ambient = rng.uniform(0.6, 1.0, size=3).astype(np.float32)
        N += (1.0 - K)[..., None] * ambient

    if glare and rng.random() < 0.5:
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        sigma = rng.uniform(2.0, 10.0)
        amp = rng.uniform(0.0, 1.0)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        tint = rng.uniform(0.8, 1.0, size=3).astype(np.float32)
        N += (amp * blob)[..., None] * tint

    # Sensor noise is not carried by the smooth field: its *realisation* is
    # high-frequency and, measured in phase 0, a field fitted to one image's
    # noise transfers nothing. Its *magnitude* does transfer, and that is what a
    # deployment estimator can actually recover from a single frame. So N here is
    # the structured additive part only; sigma is returned and drawn fresh.
    sigma = float(rng.uniform(0.0, 0.12)) if (noise and rng.random() < 0.5) else 0.0
    return K, N, sigma


def sample_degradation(rng: np.random.Generator, h: int = 32, w: int = 32,
                       blur: bool = True) -> tuple:
    """A full sample: field (K, N), noise scale sigma, and blur parameters.

    Kept next to :func:`sample_field` rather than folded into it so that callers
    that predate the kernel (probe, diagnostics) keep working unchanged.
    """
    K, N, sigma = sample_field(rng, h, w)
    return K, N, sigma, (sample_blur(rng) if blur else np.zeros(3, np.float32))


def add_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add fresh white noise of the given magnitude; a realisation, not a field."""
    if sigma <= 0:
        return x
    return np.clip(x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32), 0.0, 1.0)


def apply_field(x_clean: np.ndarray, K: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Compose one field onto a clean image in [0, 1]; returns [0, 1]."""
    return np.clip(K[..., None] * x_clean + N, 0.0, 1.0)


def compose(clean, fields):
    """Apply fields in order. This is what makes compound conditions reachable:
    two sampled fields give a joint operating point, in field space, not a
    stronger version of one operator."""
    out = clean
    for K, N in fields:
        out = apply_field(out, K, N)
    return out


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse <= 0 else 10.0 * np.log10(1.0 / mse)
