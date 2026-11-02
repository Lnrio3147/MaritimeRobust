"""Phase 1 of docs/MODULE_NCF.md: estimate a corruption field from one image.

The phase-0 probe (``scripts/cifar_field_probe.py``) showed the representation is
expressive but content-entangled: fitting (K, N) per image with the clean image
known reaches ~40 dB in sample, and applying that field to a different image is
*worse* than doing nothing. A field fitted to one picture describes that picture,
not the degradation.

So the field must be estimated by something that has seen many contents under
many fields and is therefore forced to put content in the image path and
degradation in the field path. That is this network. It is trained on the
continuous prior of ``cifar_fields.sample_field``, never on the named families,
so the held-out-family test later stays clean.

The transfer evaluation is the only honest one: estimate on image *i*, apply the
estimated field to image *j*, compare with *j*'s actual degradation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout),
                         nn.SiLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout),
                         nn.SiLU(inplace=True))


class FieldEstimator(nn.Module):
    """Image -> (K, N). ``K`` in [0, 1], ``N`` unconstrained.

    The field lives on a coarse grid (``stride`` 4, so 8x8 for a 32x32 image) and
    is bilinearly upsampled. That is a capacity limit with a physical reading —
    transmission, ambient light and glare are low-frequency — and it is also the
    only thing that stops the degenerate solution structurally: a decoder that
    emits full-resolution ``N`` can copy the picture into it, and a penalty alone
    was measured to be satisfied by a constant field instead (spread ~0).
    """

    def __init__(self, width: int = 32, stride: int = 4):
        super().__init__()
        self.stride = stride
        w = width
        self.d1 = conv_block(3, w)
        self.d2 = conv_block(w, w * 2)
        self.d3 = conv_block(w * 2, w * 4)
        self.up32 = nn.ConvTranspose2d(w * 4, w * 2, 2, stride=2)
        self.b32 = conv_block(w * 4, w * 2)
        self.up21 = nn.ConvTranspose2d(w * 2, w, 2, stride=2)
        self.b21 = conv_block(w * 2, w)
        self.k_head = nn.Conv2d(w, 1, 3, padding=1)
        self.n_head = nn.Conv2d(w, 3, 3, padding=1)
        # per-image noise magnitude: white noise has no transferable realisation,
        # only a transferable scale
        self.s_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(w * 4, 1))
        # Per-image convolution, 4 numbers: (log sigma_x, log sigma_y, cos 2theta,
        # sin 2theta). Blur is not expressible as a pixel-wise gain, and measured
        # without it every blur family sits below AugMix while every photometric
        # and noise family reaches the supervised level (docs/MODULE_NCF.md 8.5).
        self.b_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.Linear(w * 4, 64), nn.SiLU(), nn.Linear(64, 4))
        nn.init.zeros_(self.n_head.weight)
        nn.init.zeros_(self.n_head.bias)
        nn.init.zeros_(self.k_head.weight)
        nn.init.constant_(self.k_head.bias, 2.0)   # sigmoid(2) ~ 0.88: start near identity
        nn.init.zeros_(self.s_head[-1].weight)
        nn.init.constant_(self.s_head[-1].bias, -2.9)   # softplus ~ 0.05, prior mean
        nn.init.zeros_(self.b_head[-1].weight)
        # A bias of -6 (softplus ~ 0.0025) has gradient 2.5e-3, and was measured
        # to leave the head parked at zero for 12k steps: predicted sigma 0.000,
        # correlation with the truth negative (scripts/cifar_field_blurdiff.py).
        # Start where softplus' is O(1) instead, at the prior's own mean width.
        nn.init.constant_(self.b_head[-1].bias, -0.2)   # softplus ~ 0.6 px
        self.b_head[-1].bias.data[2] = 0.0               # theta = pi/2, prior mean
        self.b_head[-1].bias.data[3] = 0.0

    def blur_params(self, x):
        """(sigma_x, sigma_y, theta) per image, from the same trunk features.

        ``theta`` is a sigmoid into [0, pi] rather than ``atan2(cos, sin)``: at the
        zero-initialised output ``atan2(0, 0)`` has zero gradient, so the angle
        could never leave its start.
        """
        d = self.d1(x)
        d = self.d2(F.max_pool2d(d, 2))
        d = self.d3(F.max_pool2d(d, 2))
        raw = self.b_head(d)
        sx = F.softplus(raw[:, 0])
        sy = F.softplus(raw[:, 1])
        theta = float(torch.pi) * torch.sigmoid(raw[:, 3])
        return torch.stack([sx, sy, theta], 1)

    def coarse(self, x):
        """The structured field on the coarse grid, plus the noise magnitude."""
        d1 = self.d1(x)
        d2 = self.d2(F.max_pool2d(d1, 2))
        d3 = self.d3(F.max_pool2d(d2, 2))
        u2 = self.b32(torch.cat([self.up32(d3), d2], 1))
        u1 = self.b21(torch.cat([self.up21(u2), d1], 1))
        k = torch.sigmoid(self.k_head(u1))
        n = self.n_head(u1)
        if self.stride > 1:
            grid = (max(1, x.shape[-2] // self.stride), max(1, x.shape[-1] // self.stride))
            k = F.interpolate(k, size=grid, mode="area")
            n = F.interpolate(n, size=grid, mode="area")
        return k, n, F.softplus(self.s_head(d3)).squeeze(1)

    def forward(self, x):
        k, n, _ = self.coarse(x)
        size = x.shape[-2:]
        if k.shape[-2:] != size:
            k = F.interpolate(k, size=size, mode="bilinear", align_corners=False)
            n = F.interpolate(n, size=size, mode="bilinear", align_corners=False)
        return k, n


def apply_field_torch(x_clean, k, n):
    """Compose a field onto a clean tensor batch; returns [0, 1]."""
    return (k * x_clean + n).clamp(0.0, 1.0)


def gaussian_kernels(params: torch.Tensor, size: int = 11) -> torch.Tensor:
    """(B, 3) sigma_x/sigma_y/theta -> (B, 1, size, size) normalised kernels.

    Anisotropic Gaussians are built by rotating the coordinate grid, so an
    elongated kernel is a motion streak and a round one is defocus; a zero
    sigma pair yields a delta kernel (no blur).
    """
    import numpy as np
    from .cifar_fields import gaussian_kernel
    ks = [gaussian_kernel(p.detach().cpu().numpy(), size) for p in params]
    return torch.from_numpy(np.stack(ks)).unsqueeze(1).to(params.device, torch.float32)


def apply_blur_torch(x: torch.Tensor, params: torch.Tensor, size: int = 11) -> torch.Tensor:
    """Convolve a batch with per-image kernels; same kernel on all three channels."""
    b, c, h, w = x.shape
    r = size // 2
    k = gaussian_kernels(params, size).repeat_interleave(c, 0)      # (B*C, 1, s, s)
    # channels dimension carries one image-channel per group; one kernel per group
    xp = F.pad(x.reshape(1, b * c, h, w), (r, r, r, r), mode="reflect")
    y = F.conv2d(xp, k, groups=b * c)
    return y.reshape(b, c, h, w).clamp_(0.0, 1.0)


def total_variation(t):
    return ((t[..., 1:] - t[..., :-1]).abs().mean()
            + (t[..., :, 1:] - t[..., :, :-1]).abs().mean())


def estimate_loss(estimator, x_deg, x_clean, k_true=None, n_true=None,
                  smooth: float = 1e-2) -> tuple[torch.Tensor, dict]:
    """Reconstruction loss plus smoothness; the estimator only ever sees x_deg.

    Kept for reference: this objective is *degenerate*. One equation per pixel,
    four unknowns — ``K = 0, N = x_deg`` reconstructs any target exactly, so a
    network trained on it learns to copy the input and transfer is worse than
    doing nothing (measured: `scripts/cifar_field_diagnose.py`). Use
    :func:`equivariance_loss` instead.
    """
    k, n = estimator(x_deg)
    pred = apply_field_torch(x_clean, k, n)
    rec = (pred - x_deg).abs().mean()
    loss = rec + smooth * (total_variation(k) + total_variation(n))
    metrics = {"rec": float(rec.detach())}
    if k_true is not None:
        metrics["k_err"] = float((k - k_true).abs().mean().detach())
    return loss, metrics


def equivariance_loss(estimator, d1, d2, x1, x2, smooth: float = 1e-2,
                      equiv: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Identifiable training: one field, two contents, two estimates that agree.

    ``d1`` and ``d2`` are two *different* clean images degraded by the *same*
    field. The field that explains them is unique, so requiring the two estimates
    to agree removes the copy-the-input solution: content cannot live in ``N``
    when ``N`` must be shared across two different pictures. Nothing here is
    labelled; the constraint comes from the construction, not from any family.
    """
    k1, n1 = estimator(d1)
    k2, n2 = estimator(d2)
    rec = (((apply_field_torch(x1, k1, n1) - d1).abs().mean()
            + (apply_field_torch(x2, k2, n2) - d2).abs().mean()) * 0.5)
    eq = (k1 - k2).abs().mean() + (n1 - n2).abs().mean()
    tv = 0.5 * (total_variation(k1) + total_variation(n1)
                + total_variation(k2) + total_variation(n2))
    loss = rec + equiv * eq + smooth * tv
    return loss, {"rec": float(rec.detach()), "equv": float(eq.detach()),
                  "tv": float(tv.detach())}


def supervised_loss(estimator, x_deg, x_clean, k_true, n_true, s_true=None,
                    smooth: float = 1e-3, rec_w: float = 1.0,
                    field_w: float = 1.0, b_true=None,
                    blur_w: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Regress the field that generated the sample, plus reconstruction.

    The generator knows ``(K, N)`` — it drew them — so they are supervision, and
    that names no family: the prior is physical parameters only. Comparing on the
    estimator's own coarse grid keeps the target inside its capacity instead of
    penalising it for a resolution it does not have.

    This replaces two failed objectives, both recorded in docs/MODULE_NCF.md:
    reconstruction alone collapses to ``K = 0, N = x`` (copies the input), and a
    reconstruction + equivariance penalty is satisfied by a constant field
    (spread ~0), which explains nothing.
    """
    k, n, s = estimator.coarse(x_deg)
    grid = k.shape[-2:]
    kt = F.interpolate(k_true, size=grid, mode="area") if k_true.shape[-2:] != grid else k_true
    nt = F.interpolate(n_true, size=grid, mode="area") if n_true.shape[-2:] != grid else n_true
    field = (k - kt).abs().mean() + (n - nt).abs().mean()
    metrics = {"field": float(field.detach())}
    if s_true is not None:
        s_err = (s - s_true).abs().mean()
        field = field + s_err
        metrics["sigma_err"] = float(s_err.detach())
    if b_true is not None:
        b = estimator.blur_params(x_deg)
        # sigma in pixels. The angle is only defined for an elongated kernel: on
        # the half of the prior that draws no blur at all the target angle is a
        # placeholder, and penalising it would train the head to answer 0.
        has = (b_true[:, 0] + b_true[:, 1] > 0).float()
        blur_err = ((b[:, 0] - b_true[:, 0]).abs().mean()
                    + (b[:, 1] - b_true[:, 1]).abs().mean()
                    + (has * (torch.cos(b[:, 2]) - torch.cos(b_true[:, 2])).abs()).mean()
                    + (has * (torch.sin(b[:, 2]) - torch.sin(b_true[:, 2])).abs()).mean())
        field = field + blur_w * blur_err
        metrics["blur_err"] = float(blur_err.detach())
    kb = F.interpolate(k, size=x_deg.shape[-2:], mode="bilinear", align_corners=False)
    nb = F.interpolate(n, size=x_deg.shape[-2:], mode="bilinear", align_corners=False)
    # The generator convolved the clean image before applying the field; the
    # reconstruction must undo the same chain, or it would spend the field
    # trying to explain a blur it cannot express.
    base = apply_blur_torch(x_clean, b_true) if b_true is not None else x_clean
    rec = (apply_field_torch(base, kb, nb) - x_deg).abs().mean()
    tv = total_variation(k) + total_variation(n)
    loss = field_w * field + rec_w * rec + smooth * tv
    metrics["rec"] = float(rec.detach())
    return loss, metrics


@torch.no_grad()
def identifiability(estimator, x_clean, x_deg=None) -> dict:
    """A field estimator run on an *undegraded* image must report no field.

    This is the check that the degenerate solution fails: ``K`` collapses to 0
    and ``N`` becomes the image. A correct estimator gives ``K = 1, N = 0``.

    The spread is measured on ``x_deg`` (a batch of different degradations) when
    given: a field that is the same for every input satisfies equivariance by
    ignoring the input altogether, which the spread exposes.
    """
    k, n = estimator(x_clean)
    corr = torch.corrcoef(torch.stack([n.mean(1).flatten(), x_clean.mean(1).flatten()]))[0, 1]
    out = {"k_mean": float(k.mean()), "n_abs": float(n.abs().mean()),
           "corr_n_x": float(corr), "k_spread": 0.0, "n_spread": 0.0}
    if x_deg is not None:
        # 0 means the field is a constant for every input: equivariance was
        # satisfied by ignoring the input, which costs nothing but achieves nothing
        kd, nd = estimator(x_deg)
        out["k_spread"] = float(kd.flatten(1).std(0).mean())
        out["n_spread"] = float(nd.flatten(1).std(0).mean())
    return out
