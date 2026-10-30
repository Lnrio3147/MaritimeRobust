"""Deterministic approximate image corruptions. No generated semantics or fake depth."""
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter, uniform_filter, gaussian_filter

from .common import stable_seed

RECIPE = {
    "version": "1.0",
    "haze_transmission": [0.85, 0.70, 0.55, 0.40, 0.25],
    "cloud_alpha": [0.25, 0.40, 0.55, 0.70, 0.85],
    "cloud_area": [0.15, 0.25, 0.35, 0.45, 0.55],
    "glare_strength": [0.35, 0.65, 1.0, 1.5, 2.0],
    "glare_sigma": [0.08, 0.12, 0.16, 0.20, 0.25],
    "domains": {"matched": "uniform location; 6x6 smooth field", "shifted": "edge-biased location; 11x11 field; brighter airlight"},
    "limitation": "No physical depth; no semantic vessel mask unless supplied. Center ROI is a proxy only."
}


def conditions(include_shifted=True):
    result = [{"name": "clean", "kind": "clean", "severity": 0, "domain": "matched", "placement": "random"}]
    for domain in (["matched", "shifted"] if include_shifted else ["matched"]):
        for kind in ("haze", "cloud", "glare"):
            for severity in range(1, 6):
                result.append({"name": f"{domain}_{kind}_{severity}", "kind": kind,
                               "severity": severity, "domain": domain, "placement": "random"})
        for kind in ("haze+glare", "haze+cloud"):
            for severity in (3, 5):
                result.append({"name": f"{domain}_{kind}_{severity}", "kind": kind,
                               "severity": severity, "domain": domain, "placement": "random"})
    # Controlled spatial diagnostic, same source-level seed across placements.
    for placement in ("center", "border"):
        for kind in ("cloud", "glare"):
            result.append({"name": f"matched_{kind}_3_{placement}", "kind": kind,
                           "severity": 3, "domain": "matched", "placement": placement})
    return result


def smooth_field(rng, size, grid):
    small = Image.fromarray(rng.uniform(0, 1, (grid, grid)).astype("float32"), mode="F")
    return np.asarray(small.resize(size, Image.Resampling.BICUBIC)).clip(0, 1)


def corrupt(image, kind="clean", severity=0, seed=0, domain="matched", placement="random", target_mask=None):
    if kind != "clean" and severity not in range(1, 6):
        raise ValueError("severity must be 1..5")
    if domain not in ("matched", "shifted"):
        raise ValueError(domain)
    rng = np.random.default_rng(seed)
    a = np.array(image.convert("RGB"), dtype=np.float32) / 255
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx + 0.5) / w
    yy = (yy + 0.5) / h
    proxy = (xx > .25) & (xx < .75) & (yy > .25) & (yy < .75)
    affected = np.zeros((h, w), bool)
    params = {"kind": kind, "severity": severity, "seed": int(seed), "domain": domain,
              "placement": placement, "recipe_version": RECIPE["version"]}
    if kind == "clean":
        params.update(saturated_fraction=float((a.max(2) >= 254 / 255).mean()),
                      center_proxy_affected_fraction=0.0, target_affected_fraction=0.0 if target_mask is not None else None)
        return image.convert("RGB"), params
    i = severity - 1
    for part in kind.split("+"):
        field = smooth_field(rng, (w, h), 6 if domain == "matched" else 11)
        if part == "haze":
            t0 = RECIPE["haze_transmission"][i]
            t = np.clip(t0 + .18 * (field - .5), .05, 1)
            airlight = rng.uniform(.85, .98, 3) if domain == "matched" else rng.uniform(.95, 1, 3)
            a = a * t[..., None] + airlight * (1 - t[..., None])
            affected[:] = True
            params.update(transmission_mean=float(t.mean()), airlight=airlight.tolist())
        elif part in ("cloud", "glare"):
            if placement == "center":
                cx, cy = .5, .5
            elif placement == "border":
                cx, cy = .06, .06
            elif domain == "shifted":
                cx, cy = float(rng.choice([.08, .92])), float(rng.uniform(.1, .9))
            else:
                cx, cy = rng.uniform(.1, .9, 2)
            if part == "cloud":
                # Rank a noisy smooth spatial field; exact area is controlled independently of opacity.
                spatial = field + .6 * np.exp(-((xx-cx)**2 + (yy-cy)**2) / .06)
                threshold = np.quantile(spatial, 1 - RECIPE["cloud_area"][i])
                support = spatial >= threshold
                soft_support = gaussian_filter(support.astype(np.float32), sigma=max(1,min(h,w)*.015))
                alpha = soft_support * RECIPE["cloud_alpha"][i] * (.65 + .35 * field)
                a = a * (1 - alpha[..., None]) + .98 * alpha[..., None]
                affected |= alpha > .05
                params.update(cloud_alpha_max=float(alpha.max()), cloud_area_fraction=float(support.mean()))
            else:
                sigma = RECIPE["glare_sigma"][i]
                bloom = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*sigma**2))
                gain = RECIPE["glare_strength"][i] * bloom
                a = a + gain[..., None] * np.array([1, .98, .90])
                affected |= gain > .1
                params.update(glare_strength=RECIPE["glare_strength"][i], glare_sigma=sigma)
            params[part + "_center"] = [float(cx), float(cy)]
        else:
            raise ValueError(part)
    a = np.clip(a, 0, 1)
    params.update(saturated_fraction=float((a.max(2) >= 254 / 255).mean()),
                  affected_fraction=float(affected.mean()),
                  center_proxy_affected_fraction=float(affected[proxy].mean()) if proxy.any() else None,
                  target_affected_fraction=float(affected[np.asarray(target_mask, bool)].mean())
                  if target_mask is not None and np.asarray(target_mask, bool).any() else None)
    return Image.fromarray(np.uint8(a * 255 + .5)), params


def corruption_seed(sample_id, role, kind, severity, domain="matched", base_seed=0):
    # Placement intentionally excluded for paired spatial diagnostics.
    return stable_seed(RECIPE["version"], sample_id, role, kind, severity, domain, base_seed)


def dcp(image, omega=.95, t_min=.1, patch=15, guide_radius=20):
    """Dark channel, airlight estimation, grayscale guided-filter transmission refinement."""
    a = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    dark = minimum_filter(a.min(2), size=patch, mode="nearest")
    n = max(1, int(dark.size * .001))
    indices = np.argpartition(dark.ravel(), -n)[-n:]
    candidates = a.reshape(-1, 3)[indices]
    airlight = candidates[np.argmax(candidates.sum(1))].clip(.05, 1)
    transmission = 1 - omega * minimum_filter((a / airlight).min(2), size=patch, mode="nearest")
    guide = a @ np.array([.299, .587, .114], dtype=np.float32)
    mean = lambda x: uniform_filter(x, size=2*guide_radius+1, mode="reflect")
    mi, mp = mean(guide), mean(transmission)
    coeff = (mean(guide*transmission) - mi*mp) / (mean(guide*guide) - mi*mi + 1e-3)
    intercept = mp - coeff*mi
    refined = np.clip(mean(coeff)*guide + mean(intercept), t_min, 1)
    restored = np.clip((a-airlight)/refined[...,None] + airlight, 0, 1)
    return Image.fromarray(np.uint8(restored*255+.5))
