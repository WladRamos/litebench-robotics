"""Luz da lightbox e do domo (scene.light_colors) e medição de aparência; NumPy puro."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from vials_lite import spec
from vials_lite.scene import blackbody_rgb, dome_direction, light_colors


def _measure_appearance():
    path = Path(__file__).parents[1] / "scripts" / "measure_appearance.py"
    mod_spec = importlib.util.spec_from_file_location("measure_appearance", path)
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules[mod_spec.name] = mod
    mod_spec.loader.exec_module(mod)
    return mod


def test_blackbody_is_warm_at_low_and_cool_at_high_temperature():
    warm, cool = blackbody_rgb(2500.0), blackbody_rgb(9500.0)
    assert warm.max() == cool.max() == 1.0
    assert warm[0] > warm[2] and cool[2] > cool[0]
    np.testing.assert_allclose(blackbody_rgb(6600.0), 1.0, atol=0.02)


def test_eval_has_no_dome_and_exposure_scales_the_lightbox():
    dark, bright = light_colors(-3.0), light_colors(1.0)
    assert not dark["dome"].any() and not bright["dome"].any()
    ratio = bright["rect"][0] / dark["rect"][0]
    np.testing.assert_allclose(ratio, 2.0 ** (4 * spec.LIGHTING.exposure_scale))
    np.testing.assert_allclose(bright["ambient"] / dark["ambient"], ratio)


def test_dome_temperature_tints_and_white_point_is_neutral():
    calib = spec.LIGHTING
    white = light_colors(0.0, 0.0, calib.white_k)["dome"]
    np.testing.assert_allclose(white, white[0])
    warm = light_colors(0.0, 0.0, 2500.0)["dome"]
    cool = light_colors(0.0, 0.0, 9500.0 + 3000.0)["dome"]
    assert warm[0] > warm[2] and cool[2] > cool[0]
    assert np.all(warm >= 0) and np.all(cool >= 0)


def test_dome_enters_through_the_open_front_going_down():
    d = dome_direction(spec.LIGHTING.dome_elevation_deg)
    np.testing.assert_allclose(np.linalg.norm(d), 1.0)
    assert d[0] > 0 and d[2] <= 0 and d[1] == 0


def test_region_means_trim_outliers_and_keep_dark_half_on_mat():
    region_means = _measure_appearance().region_means
    img = np.full((4, 5, 3), 100.0)
    img[0, 0] = 255.0  # reflexo isolado: sai na média aparada
    mat = np.zeros((4, 5, 3))
    mat[:, :2] = 200.0  # rack amarelo invadindo a máscara do tapete
    masks = {"floor": np.ones((4, 5), bool), "mat": np.ones((4, 5), bool)}
    out = region_means({"default": img, "mat": mat}, masks)
    np.testing.assert_allclose(out["floor"], 100.0)
    np.testing.assert_allclose(out["mat"], 0.0)
