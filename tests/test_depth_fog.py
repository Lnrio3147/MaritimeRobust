import numpy as np
import pytest
from PIL import Image

from maritime_robust import depth_fog as df


def gradient_image(h=64, w=96):
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([x / w, y / h, (x + y) / (h + w)], -1)
    return Image.fromarray((a * 255).astype(np.uint8))


def test_flat_depth_reproduces_far_transmission_grid():
    d = np.ones((8, 8), np.float32)
    for severity, t_far in zip(range(1, 6), df.DEPTH_FOG_RECIPE["far_transmission"]):
        t = df.transmission_map(d, severity, "matched", turbulence=0)
        assert np.allclose(t, t_far, atol=1e-5)


def test_transmission_decreases_with_depth():
    d = np.linspace(0, 1, 11, dtype=np.float32)[None, :].repeat(4, 0)
    t = df.transmission_map(d, 3, "matched", turbulence=0)
    assert np.all(np.diff(t, axis=1) < 0)


def test_severity_is_monotonic():
    d = np.random.default_rng(0).uniform(0, 1, (16, 16)).astype(np.float32)
    means = [df.transmission_map(d, s, turbulence=0).mean() for s in range(1, 6)]
    assert all(b < a for a, b in zip(means, means[1:]))


def test_fog_is_deterministic_and_keeps_shape():
    img, d = gradient_image(), np.random.default_rng(1).uniform(0, 1, (64, 96)).astype(np.float32)
    a, _ = df.apply_fog(img, d, 5, seed=7)
    b, _ = df.apply_fog(img, d, 5, seed=7)
    assert a.size == img.size and np.array_equal(np.asarray(a), np.asarray(b))


def test_far_field_is_more_fogged_than_near():
    d = np.zeros((32, 32), np.float32)
    d[:, 16:] = 1.0
    img = Image.new("RGB", (32, 32), (90, 90, 90))
    out, params = df.apply_fog(img, d, 4, seed=3, turbulence=0)
    arr = np.asarray(out, np.float32)
    assert arr[:, :16].mean() < arr[:, 16:].mean()
    assert params["transmission_min"] < params["transmission_max"]


def test_corrupt_with_depth_contract():
    img, d = gradient_image(), np.ones((64, 96), np.float32)
    out, params = df.corrupt_with_depth(img, d, "depth_haze+glare", 3, seed=5)
    assert out.size == img.size
    assert params["kind"] == "depth_haze+glare" and "glare_strength" in params
    clean, _ = df.corrupt_with_depth(img, d, "clean")
    assert np.array_equal(np.asarray(clean), np.asarray(img))


def test_corrupt_with_depth_rejects_bad_kind_and_severity():
    img, d = gradient_image(), np.ones((64, 96), np.float32)
    for severity in (0, 6):
        with pytest.raises(ValueError):
            df.corrupt_with_depth(img, d, "depth_haze", severity)
    with pytest.raises(ValueError):
        df.corrupt_with_depth(img, d, "haze", 3)


def test_horizon_provider_is_deterministic_and_ordered():
    provider = df.DepthProvider("horizon", horizon=0.4)
    d = provider(gradient_image(50, 80))
    assert d.shape == (50, 80) and 0 <= d.min() <= d.max() <= 1
    assert np.array_equal(d, provider(gradient_image(50, 80)))
    assert np.isclose(d[0].mean(), 1.0) and d[-1].mean() < d[-2].mean()


def test_conditions_are_unique():
    names = [c["name"] for c in df.fog_conditions()]
    assert len(names) == len(set(names)) == 18
