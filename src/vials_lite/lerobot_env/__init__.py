"""Integração com o LeRobot 0.6.1: o Vials-Lite como ambiente do ``lerobot-eval``.

Dois caminhos, o mesmo ``make_env``:

* **plugin local** (sem Hub): este módulo registra o tipo de ambiente ``vials_lite``::

      lerobot-eval --env.discover_packages_path=vials_lite.lerobot_env --env.type=vials_lite
          --env.variant=eval --policy.path=<checkpoint> --eval.n_episodes=10 --eval.batch_size=1

  (``--env.layout=dataset``: o arranjo da cena do dataset do curso;
  ``--env.init_pose=dataset`` e ``--env.external_cam=fixed``: pose inicial dos operadores e câmera
  externa sem sorteio.)

* **EnvHub**: ``envhub/env.py`` reexporta ``make_env``; publicado num repositório do Hub, vale
  ``lerobot-eval --env.type=... `` com ``hub_path`` ou ``make_env("<user>/<repo>",
  trust_remote_code=True)``. A máquina que avalia precisa ter o pacote ``vials_lite`` instalado.

Observações no formato do ``lerobot-eval`` (``envs.utils.preprocess_observation``): ``agent_pos`` →
``observation.state``; ``pixels/ego`` e ``pixels/external_D455`` → ``observation.images.ego`` e
``observation.images.external_D455``, as mesmas chaves do dataset do curso. Ações no espaço do
LeRobot (−100..100; garra 0..100), como no dataset.

Só importe este módulo com o extra ``lerobot`` instalado; o resto do pacote não depende dele.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .. import spec
from ..env import EXTERNAL_CAMS, INIT_POSES, LAYOUTS, VARIANTS, VialsLiteEnv
from ..scene import ACTUATOR_PRESETS, IMAGE_HEIGHT, IMAGE_WIDTH

ENV_TYPE = "vials_lite"
CAMERAS = ("ego", "external_D455")


def _features() -> dict[str, PolicyFeature]:
    feats = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    for cam in CAMERAS:
        feats[f"pixels/{cam}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    return feats


def _features_map() -> dict[str, str]:
    fmap = {ACTION: ACTION, "agent_pos": OBS_STATE}
    for cam in CAMERAS:
        fmap[f"pixels/{cam}"] = f"{OBS_IMAGES}.{cam}"
    return fmap


@EnvConfig.register_subclass(ENV_TYPE)
@dataclass
class VialsLiteEnvConfig(EnvConfig):
    """Configuração do ``lerobot-eval`` (``--env.type=vials_lite``)."""

    task: str | None = spec.LANGUAGE_INSTRUCTION
    fps: int = round(spec.TIMING.control_hz)  # 60: um step por frame de vídeo, em tempo real
    variant: str = "eval"
    layout: str = "workshop"
    init_pose: str = "workshop"
    external_cam: str = "random"
    actuators: str = "isaac"
    features: dict[str, PolicyFeature] = field(default_factory=_features)
    features_map: dict[str, str] = field(default_factory=_features_map)

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant deve ser um de {VARIANTS}, não {self.variant!r}")
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout deve ser um de {LAYOUTS}, não {self.layout!r}")
        if self.init_pose not in INIT_POSES:
            raise ValueError(f"init_pose deve ser um de {INIT_POSES}, não {self.init_pose!r}")
        if self.external_cam not in EXTERNAL_CAMS:
            raise ValueError(f"external_cam deve ser um de {EXTERNAL_CAMS}, não {self.external_cam!r}")
        if self.actuators not in ACTUATOR_PRESETS:
            raise ValueError(f"actuators deve ser um de {ACTUATOR_PRESETS}, não {self.actuators!r}")

    @property
    def gym_kwargs(self) -> dict:
        return {"variant": self.variant, "layout": self.layout, "init_pose": self.init_pose,
                "external_cam": self.external_cam, "actuators": self.actuators,
                "render_mode": "rgb_array"}

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        return make_env(n_envs=n_envs, use_async_envs=use_async_envs, cfg=self)


class VectorInfoWrapper(gym.Wrapper):
    """Deixa ``info["privileged"]`` empilhável pelo ``VectorEnv``: ``None`` vira −1 e a lista de
    eventos vira uma string separada por "|" ("" sem eventos)."""

    @staticmethod
    def _clean(info: dict) -> dict:
        p = dict(info.get("privileged", {}))
        if p.get("vial_on_rack_at_reset") is None:
            p["vial_on_rack_at_reset"] = -1
        p["events"] = "|".join(p.get("events", []))
        return {**info, "privileged": p}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._clean(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._clean(info)


def make_env(n_envs: int = 1, use_async_envs: bool = False, cfg: EnvConfig | None = None):
    """``{"vials_lite": {0: VectorEnv}}``, a forma que o ``lerobot-eval`` e o EnvHub esperam.

    ``cfg`` (opcional) fornece ``variant``, ``layout``, ``init_pose``, ``external_cam``, ``actuators``
    e ``task``; sem ele, a variante
    ``eval`` no arranjo do workshop.
    O reset automático é no mesmo step (``SAME_STEP``), como nos ambientes do próprio LeRobot.
    """
    if n_envs < 1:
        raise ValueError("n_envs deve ser ≥ 1")
    kwargs = {"variant": "eval", "actuators": "isaac", "render_mode": "rgb_array"}
    if cfg is not None:
        keys = ("variant", "layout", "init_pose", "external_cam", "actuators", "task")
        kwargs.update({k: getattr(cfg, k) for k in keys if hasattr(cfg, k)})

    def _make_one():
        return VectorInfoWrapper(VialsLiteEnv(**kwargs))

    env_cls = gym.vector.AsyncVectorEnv if (use_async_envs and n_envs > 1) else gym.vector.SyncVectorEnv
    extra = {"context": "spawn"} if env_cls is gym.vector.AsyncVectorEnv else {}
    vec = env_cls(
        [_make_one for _ in range(n_envs)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP, **extra
    )
    return {ENV_TYPE: {0: vec}}
