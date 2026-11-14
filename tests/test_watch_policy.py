"""``scripts/watch_policy.py`` sem janela: a demo roteirizada como política, o ritmo do viewer e o modo
navegador (``--stream``)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from vials_lite.scene import default_menagerie_dir  # noqa: E402


def _load_script():
    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("watch_policy", scripts / "watch_policy.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeViewer:
    """Conta as sincronizações e fecha depois de ``close_after`` delas."""

    def __init__(self, close_after: int | None = None):
        self.syncs, self.close_after = 0, close_after

    def is_running(self) -> bool:
        return self.close_after is None or self.syncs < self.close_after

    def sync(self) -> None:
        self.syncs += 1


def _sync(mod, viewer):
    sync = mod.ViewerSync(speed=1e6)  # sem esperar o tempo real
    sync.viewer = viewer
    return sync


def test_closed_viewer_stops_the_episode():
    mod = _load_script()
    sync = _sync(mod, FakeViewer(close_after=2))
    sync.append(None)
    sync.append(None)
    with pytest.raises(mod.ViewerClosed):
        sync.append(None)


@pytest.mark.skipif(
    not (default_menagerie_dir() / "robotstudio_so101" / "so101.xml").is_file(),
    reason="MuJoCo Menagerie ausente (ver assets/README.md)",
)
def test_scripted_policy_runs_an_episode_through_the_runner():
    mod = _load_script()
    env = mod.VialsLiteEnv(variant="eval", render_images=False)
    viewer = FakeViewer()
    ep = mod.run_episode(env, mod.ScriptedPolicy(env), seed=1, warmup_steps=0, frames=_sync(mod, viewer))
    env.close()
    assert ep["success"]
    assert viewer.syncs == ep["steps"]  # uma sincronização por step


@pytest.mark.skipif(
    not (default_menagerie_dir() / "robotstudio_so101" / "so101.xml").is_file(),
    reason="MuJoCo Menagerie ausente (ver assets/README.md)",
)
def test_stream_sink_renders_the_camera_chosen_by_the_page():
    mod = _load_script()
    env = mod.VialsLiteEnv(variant="eval", render_images=False)
    env.reset(seed=1)
    sink = mod.StreamSink(env, speed=1e6, camera="external_D455")
    try:
        assert sink.cameras[-1] == mod.FREE_CAMERA and "ego" in sink.cameras
        sink.append(None)
        assert sink.frame is None  # renderiza um step sim, um não
        sink.append(None)
        external = sink.frame.copy()
        sink.control({"camera": [mod.FREE_CAMERA], "orbit": ["10,5"], "zoom": ["1"]})
        sink.append(None)
        sink.append(None)
        assert sink.status()["camera"] == mod.FREE_CAMERA
        assert sink.frame.shape == (480, 640, 3) and not np.array_equal(sink.frame, external)
        sink.control({"camera": ["não existe"]})
        assert sink.camera == mod.FREE_CAMERA
    finally:
        sink.close()
        env.close()
