"""Depth-conditioned atmospheric fog.

`degradations.py` is byte-identical to the file referenced by every combined-augmentation run
config and re-hashed by `scripts/audit_delivery.py`, so the depth-aware variant lives here
instead of editing it. Physical model: I = J * t + A * (1 - t) with t = exp(-beta * d),
d normalized to [near_depth, 1]; beta is set so that the far field transmittance equals the
same 5-level grid used by the flat haze recipe, which keeps the two comparable.
"""
import numpy as np
from PIL import Image

from .common import stable_seed
from .degradations import corrupt as corrupt_flat
from .degradations import smooth_field

DEPTH_FOG_RECIPE = {
    "version": "1.0",
    "far_transmission": [0.85, 0.70, 0.55, 0.40, 0.25],
    "near_depth": 0.15,
    "turbulence": {"matched": 0.12, "shifted": 0.22},
    "airlight": {"matched": (0.80, 0.95), "shifted": (0.90, 1.00)},
    "airlight_percentile": 95.0,
    "horizon": 0.5,
    "providers": ("flat", "horizon", "depth-anything"),
}
DEPTH_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"


def fog_conditions(include_shifted=True):
    result = []
    for domain in (["matched", "shifted"] if include_shifted else ["matched"]):
        for severity in range(1, 6):
            result.append({"name": f"{domain}_depth_haze_{severity}", "kind": "depth_haze",
                           "severity": severity, "domain": domain, "placement": "random"})
        for kind in ("depth_haze+glare", "depth_haze+cloud"):
            for severity in (3, 5):
                result.append({"name": f"{domain}_{kind}_{severity}", "kind": kind,
                               "severity": severity, "domain": domain, "placement": "random"})
    return result


def depth_fog_seed(sample_id, role, kind, severity, domain="matched", base_seed=0):
    return stable_seed(DEPTH_FOG_RECIPE["version"], sample_id, role, kind, severity, domain, base_seed)


def transmission_map(depth, severity, domain="matched", field=None, turbulence=None):
    """t = exp(-beta * d); depth 1 = farthest, matching the flat recipe's far transmittance."""
    if severity not in range(1, 6):
        raise ValueError("severity must be 1..5")
    if domain not in ("matched", "shifted"):
        raise ValueError(domain)
    t_far = DEPTH_FOG_RECIPE["far_transmission"][severity - 1]
    near = DEPTH_FOG_RECIPE["near_depth"]
    d = near + np.clip(np.asarray(depth, dtype=np.float32), 0, 1) * (1 - near)
    if turbulence is None:
        turbulence = DEPTH_FOG_RECIPE["turbulence"][domain]
    if turbulence and field is not None:
        d = d * (1 + turbulence * (np.asarray(field, np.float32) - .5) * 2)
        d = np.clip(d, 1e-3, None)
    return np.exp(np.log(t_far) * d).astype(np.float32)


def airlight(a, domain, rng):
    lo, hi = DEPTH_FOG_RECIPE["airlight"][domain]
    target = rng.uniform(lo, hi)
    bright = np.percentile(a.reshape(-1, 3), DEPTH_FOG_RECIPE["airlight_percentile"], axis=0)
    return np.clip(.5 * bright + .5 * target, .55, 1.0).astype(np.float32)


def apply_fog(image, depth, severity, seed=0, domain="matched", turbulence=None, t_override=None):
    """`t_override` supplies a fixed transmission map (equal-mean control); depth is then unused."""
    if domain not in ("matched", "shifted"):
        raise ValueError(domain)
    rng = np.random.default_rng(seed)
    a = np.array(image.convert("RGB"), dtype=np.float32) / 255
    h, w = a.shape[:2]
    field = smooth_field(rng, (w, h), 6 if domain == "matched" else 11)
    t = transmission_map(depth, severity, domain, field=field, turbulence=turbulence) \
        if t_override is None else np.broadcast_to(np.asarray(t_override, np.float32), (h, w))
    A = airlight(a, domain, rng)
    out = np.clip(a * t[..., None] + A * (1 - t[..., None]), 0, 1)
    params = {"transmission_mean": float(t.mean()), "transmission_min": float(t.min()),
              "transmission_max": float(t.max()), "airlight": [float(x) for x in A],
              "airlight_gray": float(A.mean()), "depth_mean": float(np.mean(depth))}
    return Image.fromarray(np.uint8(out * 255 + .5)), params


def corrupt_with_depth(image, depth, kind="depth_haze", severity=0, seed=0, domain="matched",
                       placement="random", target_mask=None, turbulence=None):
    """Same contract as `degradations.corrupt`, plus a depth map for the fog part.

    Composite kinds apply fog first and then the flat glare/cloud part, so the frozen
    implementations stay the single source of truth for those two corruptions.
    """
    if kind == "clean":
        return corrupt_flat(image, kind="clean", severity=0, seed=seed, domain=domain,
                            placement=placement, target_mask=target_mask)
    if severity not in range(1, 6):
        raise ValueError("severity must be 1..5")
    parts = kind.split("+")
    if "depth_haze" not in parts or len(parts) < 2:
        raise ValueError(f"unsupported kind {kind}")
    fogged, params = apply_fog(image, depth, severity, seed=seed, domain=domain, turbulence=turbulence)
    params.update(kind=kind, severity=severity, seed=int(seed), domain=domain,
                  placement=placement, recipe_version=DEPTH_FOG_RECIPE["version"])
    out = fogged
    for part in parts:
        if part == "depth_haze":
            continue
        out, extra = corrupt_flat(out, kind=part, severity=severity, seed=stable_seed(seed, part),
                                  domain=domain, placement=placement, target_mask=target_mask)
        meta = {"kind", "severity", "seed", "domain", "placement", "recipe_version"}
        params.update({k: v for k, v in extra.items() if k not in meta})
    a = np.asarray(out, dtype=np.float32) / 255
    params.update(saturated_fraction=float((a.max(2) >= 254 / 255).mean()),
                  affected_fraction=1.0 if parts else 0.0)
    return out, params


def save_cached_depth(directory, sample_id, depth, size=(224, 224)):
    """Store at a fixed square size; the same image always maps back to its own aspect."""
    from pathlib import Path
    path = Path(directory) / f"{sample_id}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    field = Image.fromarray(np.asarray(depth, np.float32), mode="F").resize(size, Image.Resampling.BILINEAR)
    np.save(path, np.asarray(field, np.float16))
    return path


def load_cached_depth(directory, sample_id, size):
    """`size` is (width, height) of the image the fog will be applied to."""
    from pathlib import Path
    path = Path(directory) / f"{sample_id}.npy"
    field = np.load(path).astype(np.float32)
    if field.shape != (size[1], size[0]):
        field = np.asarray(Image.fromarray(field, mode="F").resize(size, Image.Resampling.BILINEAR),
                           np.float32)
    return field


_DEPTH_INPUT_SIZE = 518  # Depth Anything V2 default; multiple of 14


def _depth_input(image, size=_DEPTH_INPUT_SIZE):
    import torch
    w, h = image.size
    scale = size / min(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    a = np.asarray(image.resize((nw, nh), Image.Resampling.BICUBIC), np.float32) / 255
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(((a - mean) / std).transpose(2, 0, 1))[None]


class DepthProvider:
    """Monocular depth, returned as normalized depth with 1 = farthest.

    `flat` is the depth-blind control (every pixel at max depth, i.e. the flat haze),
    `horizon` is an offline geometric prior, `depth-anything` loads a HF checkpoint.
    """

    def __init__(self, kind="depth-anything", device=None, horizon=None, checkpoint=DEPTH_CHECKPOINT):
        if kind not in DEPTH_FOG_RECIPE["providers"]:
            raise ValueError(kind)
        self.kind = kind
        self.horizon = DEPTH_FOG_RECIPE["horizon"] if horizon is None else float(horizon)
        self.checkpoint = checkpoint
        self._device = device
        self._model = self._processor = None

    def _load(self):
        import torch
        from transformers import AutoModelForDepthEstimation
        self._device = torch.device(self._device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # The repo ships a pre-5.x processor config; preprocessing is done manually below.
        self._model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint).to(self._device).eval()
        return torch

    def __call__(self, image):
        return self.depth(image)

    def depth(self, image):
        rgb = image.convert("RGB")
        h, w = rgb.size[1], rgb.size[0]
        if self.kind == "flat":
            return np.ones((h, w), np.float32)
        if self.kind == "horizon":
            y = (np.arange(h, dtype=np.float32) + .5) / h
            yh = min(max(self.horizon, 1e-3), 1 - 1e-3)
            sea = np.clip((1 - y) / (1 - yh), 0, 1)
            field = np.where(y <= yh, 1.0, sea).astype(np.float32)
            return np.repeat(field[:, None], w, axis=1)
        if self._model is None:
            self._load()
        import torch
        pixels = _depth_input(rgb).to(self._device)
        with torch.no_grad():
            pred = self._model(pixel_values=pixels).predicted_depth
        pred = torch.nn.functional.interpolate(pred[None], size=(h, w), mode="bicubic",
                                               align_corners=False)[0, 0]
        d = pred.detach().cpu().numpy().astype(np.float32)
        lo, hi = np.percentile(d, [2, 98])
        disparity = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        return (1 - disparity).astype(np.float32)

    def release(self):
        self._model = self._processor = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
