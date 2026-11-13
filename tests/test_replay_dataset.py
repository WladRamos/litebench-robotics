import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_script():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("replay_dataset", scripts / "replay_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_held_interval_picks_longest_blocked_stretch():
    held_interval = _load_script().held_interval
    # Como no episódio 0: abre (23), fecha (comando 1, estado trava em 18), solta (28).
    action = np.array([10, 23, 23, 1, 1, 1, 1, 28, 28, 1, 28], dtype=float)
    state = np.array([10, 14, 23, 18, 18, 18, 18, 19, 28, 18, 28], dtype=float)
    assert held_interval(state, action) == (3, 7)
    assert held_interval(action, action) is None


def test_tracking_summary_respects_mask():
    spec = importlib.util.spec_from_file_location(
        "replay_dynamic", Path(__file__).parents[1] / "scripts" / "replay_dynamic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    _load_script()  # replay_dynamic importa replay_dataset
    spec.loader.exec_module(mod)
    err = np.array([[1.0, 10.0], [-3.0, 0.5]])
    mask = np.array([[True, False], [True, True]])
    s = mod.tracking_summary(err, mask)
    np.testing.assert_allclose(s["mean"], [2.0, 0.5])
    np.testing.assert_allclose(s["max"], [3.0, 0.5])
    np.testing.assert_allclose(s["frac_ok"], [0.5, 1.0])


def _load_fit_camera():
    spec = importlib.util.spec_from_file_location(
        "fit_external_camera", Path(__file__).parents[1] / "scripts" / "fit_external_camera.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_camera_fit_change_masks_agree_for_a_moving_block():
    fit = _load_fit_camera()
    # um bloco claro que anda sobre fundo escuro, 5 frames: observado e previsto devem coincidir
    sil = np.zeros((5, 8, 10), dtype=bool)
    for k in range(5):
        sil[k, 2:5, 2 * k : 2 * k + 2] = True
    frames = np.where(sil[..., None], 200, 20).astype(np.uint8).repeat(3, axis=-1)
    observed = fit.change_mask(frames)
    np.testing.assert_array_equal(observed, fit.predicted_change(sil))
    assert fit.iou(observed, fit.predicted_change(sil)) == 1.0
    assert fit.iou(observed, np.zeros_like(observed)) == 0.0
