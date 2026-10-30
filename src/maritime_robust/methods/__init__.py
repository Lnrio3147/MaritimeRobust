"""S1 candidate methods.

New code only: nothing here modifies ``degradations.py``, ``engine.py``,
``cli.py`` or the combined-augmentation recipe, all of which are hash-bound to
existing runs.

Every method is expressed as the same wrapper around a pretrained backbone, so
that the "none" arm and the candidate arms differ in exactly one place.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from maritime_robust.engine import build_model

DIMS = {"convnext_tiny": 768, "swin_t": 768, "resnet50": 2048}
METHODS = ("none", "r3", "i1", "t1", "i2", "r1", "t2", "t3", "m1", "d1", "d1inv")
CLIP = 0.99
CG_MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
CG_STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def clip_mask(x):
    """M1: per-pixel mask of censored (sensor-clipped) evidence.

    ``x`` is a ``[0, 1]`` image tensor; a pixel counts as censored as soon as any
    channel reaches the clip ceiling, because the underlying scene value is then
    unobservable rather than merely bright.
    """
    return (x.amax(x.dim() - 3, keepdim=True) >= CLIP).to(x.dtype)


def retinex(x, kernel=31):
    """I2: single-scale Retinex band-pass on the [0,1] image, re-normalised."""
    mean = CG_MEAN.to(x.device)
    std = CG_STD.to(x.device)
    img = (x * std + mean).clamp(1e-3, 1 - 1e-3)
    illumination = F.avg_pool2d(img, kernel, stride=1, padding=kernel // 2)
    band = img.log() - illumination.log()
    band = (band - band.mean((2, 3), keepdim=True)) / (band.std((2, 3), keepdim=True) + 1e-6)
    return band * std + mean


class Denoise(nn.Module):
    """R3: residual channel denoising on the pooled feature."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class Descriptor(nn.Module):
    """Hand-computed image statistics used as a degradation descriptor.

    12 values: grey mean/std, mean |grad| x/y, three channel means, three
    channel stds, and the red/blue channel stds.
    """

    out_dim = 12

    def forward(self, x):
        grey = x.mean(1, keepdim=True)
        gx = (grey[:, :, :, 1:] - grey[:, :, :, :-1]).abs().mean((2, 3))
        gy = (grey[:, :, 1:, :] - grey[:, :, :-1, :]).abs().mean((2, 3))
        feats = [grey.mean((2, 3)), grey.std((2, 3)), gx, gy,
                 x.mean((2, 3)), x.std((2, 3)),
                 x[:, 0].std((1, 2)).unsqueeze(1), x[:, 2].std((1, 2)).unsqueeze(1)]
        return torch.cat(feats, 1)


class MethodModel(nn.Module):
    """Backbone + one intervention. ``method="none"`` is the plain classifier."""

    def __init__(self, model_name, num_classes, method="none"):
        super().__init__()
        # p1 is the physical-renderer *recipe* arm: same network as "none"
        if method == "p1":
            method = "none"
        if method not in METHODS:
            raise ValueError(method)
        base = build_model(model_name, num_classes, pretrained=True)
        if hasattr(base, "features") and hasattr(base, "classifier"):
            # ConvNeXt: ``features`` is (B,C,H,W) and ``classifier[0]`` is the
            # final LayerNorm, so the split at features / norm is native.
            self.features = base.features
            self.avgpool = base.avgpool
            self.norm = base.classifier[0]
        elif hasattr(base, "layer4"):
            # ResNet: the penultimate feature map is everything up to layer4,
            # and the classifier applies avgpool -> flatten -> fc with no norm.
            self.features = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                          base.layer1, base.layer2, base.layer3,
                                          base.layer4)
            self.avgpool = base.avgpool
            self.norm = nn.Identity()
        else:
            raise ValueError(f"unsupported backbone {model_name}")
        dim = DIMS[model_name]
        if method in ("r1", "m1") and model_name != "convnext_tiny":
            raise ValueError(f"method {method} is written against ConvNeXt internals")
        self.method = method
        self.num_classes = num_classes
        if method == "r1":
            # ConvNeXt-T features: 0 stem, then (downsample, blocks) per stage;
            # block-stage outputs are at 1/3/5/7 with 96/192/384/768 channels.
            self.stage_idx = (1, 3, 5, 7)
            self.stage_norm = nn.ModuleList([nn.LayerNorm(c) for c in (96, 192, 384, 768)])
            self.stage_proj = nn.ModuleList([nn.Linear(c, dim) for c in (96, 192, 384, 768)])
            self.fuse_norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, num_classes)
        elif method == "r3":
            self.head = nn.Sequential(Denoise(dim), nn.Linear(dim, num_classes))
        elif method == "i1":
            self.desc = Descriptor()
            self.film = nn.Sequential(nn.Linear(Descriptor.out_dim, dim), nn.GELU(),
                                      nn.Linear(dim, 2 * dim))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
            self.head = nn.Linear(dim, num_classes)
        else:
            self.head = nn.Linear(dim, num_classes)
        if method == "m1":
            # Widen the patchify stem to take the mask as a fourth channel. The
            # new weights start at zero, so the arm begins numerically identical
            # to the "none" arm and the mask has to earn its gradient.
            stem = self.features[0][0]
            conv = nn.Conv2d(stem.in_channels + 1, stem.out_channels, stem.kernel_size,
                             stem.stride, stem.padding, bias=stem.bias is not None)
            with torch.no_grad():
                conv.weight.zero_()
                conv.weight[:, :stem.in_channels].copy_(stem.weight)
                if stem.bias is not None:
                    conv.bias.copy_(stem.bias)
            self.features[0][0] = conv

    def features_of(self, x):
        if self.method == "m1":
            feat = self.features(x)
            # Classify from uncensored evidence only: average the feature map
            # over the pixels that were not clipped, with a small floor so a
            # fully censored image still yields a (weak) descriptor.
            weight = 1.0 - F.adaptive_avg_pool2d(x[:, 3:4].clamp(0, 1), feat.shape[-2:])
            weight = weight + 0.05
            pooled = ((feat * weight).sum((2, 3), keepdim=True)
                      / weight.sum((2, 3), keepdim=True))
            return torch.flatten(self.norm(pooled), 1)
        return torch.flatten(self.norm(self.avgpool(self.features(x))), 1)

    def forward(self, x):
        if self.method == "i2":
            x = retinex(x)
        if self.method == "r1":
            pooled = []
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self.stage_idx:
                    pooled.append(x.mean((2, 3)))
            f = sum(self.stage_proj[j](self.stage_norm[j](p)) for j, p in enumerate(pooled))
            return self.head(self.fuse_norm(f))
        f = self.features_of(x)
        if self.method == "i1":
            gamma, beta = self.film(self.desc(x)).chunk(2, -1)
            f = f * (1.0 + gamma) + beta
        return self.head(f)


def build(model_name, num_classes, method="none"):
    return MethodModel(model_name, num_classes, method)


TIMM_22K = "convnext_tiny.fb_in22k_ft_in1k"


def _remap_timm(src):
    """timm ConvNeXt state dict -> this module's MethodModel key space."""
    out = {}
    tails = [("conv_dw.", "block.0."), ("norm.", "block.2."),
             ("mlp.fc1.", "block.3."), ("mlp.fc2.", "block.5.")]
    for key, value in src.items():
        if key.startswith("stem."):
            out["features.0." + key[len("stem."):]] = value
        elif key.startswith("stages."):
            _, stage, rest = key.split(".", 2)
            stage = int(stage)
            if rest.startswith("blocks."):
                _, block, tail = rest.split(".", 2)
                base = f"features.{1 + 2 * stage}.{block}."
                for prefix, target in tails:
                    if tail.startswith(prefix):
                        out[base + target + tail[len(prefix):]] = value
                        break
                else:
                    if tail == "gamma":
                        out[base + "layer_scale"] = value.reshape(-1, 1, 1)
            elif rest.startswith("downsample."):
                _, layer, tail = rest.split(".", 2)
                out[f"features.{2 * stage}.{layer}." + tail] = value
        elif key.startswith("head.norm."):
            out["norm." + key[len("head.norm."):]] = value.reshape(-1)
    return out


def load_imagenet22k(model):
    """Overwrite a ConvNeXt-T MethodModel with timm's 22K-then-1K weights."""
    import timm
    reference = timm.create_model(TIMM_22K, pretrained=True, num_classes=1000)
    mapped = _remap_timm(reference.state_dict())
    target = model.state_dict()
    usable = {k: v for k, v in mapped.items()
              if k in target and tuple(target[k].shape) == tuple(v.shape)}
    missing = [k for k in target if k not in usable and not k.startswith("head.")]
    model.load_state_dict(usable, strict=False)
    return {"loaded": len(usable), "unmatched_target_keys": missing,
            "total_target": len(target)}
