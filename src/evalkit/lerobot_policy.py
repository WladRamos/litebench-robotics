"""Backend de política do LeRobot: ``from_pretrained`` + ``select_action``.

Carrega qualquer política salva pelo LeRobot 0.6.1 (pasta local ou repo do Hub) com os seus
pré e pós-processadores (normalização), e adapta a observação do ``VialsLiteEnv``
(``agent_pos`` e ``pixels`` em uint8 HWC) para as chaves do dataset do curso. Políticas com
action chunking guardam a fila no próprio ``select_action``, então ``act`` devolve uma ação por
step, como no ``lerobot_eval.py`` do workshop.

pi0.5 numa GPU de 8 GB: o modelo tem 4,1 bilhões de parâmetros (7,9 GB em bf16,
com o encoder de visão em fp32) e não cabe. Ele é carregado na CPU, perde as duas ``lm_head`` (geram
texto; a previsão de ações não passa por elas) e a tabela de embeddings do texto fica na CPU (uma
consulta por token, mesmo resultado); o resto (6,3 GiB) vai para a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _slim_pi05(policy) -> None:
    """Tira da GPU o que o pi0.5 não usa para prever ações (ver a docstring do módulo)."""
    import torch

    pwe = policy.model.paligemma_with_expert
    pwe.paligemma.lm_head = None
    pwe.gemma_expert.lm_head = None
    lm = pwe.paligemma.model.language_model

    class CpuEmbedding(torch.nn.Module):
        def __init__(self, embedding: torch.nn.Module) -> None:
            super().__init__()
            self._embedding = [embedding]  # numa lista, para o .to(device) não levá-la à GPU

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self._embedding[0](ids.to("cpu")).to(ids.device)

    lm.embed_tokens = CpuEmbedding(lm.embed_tokens)


class LeRobotPolicy:
    def __init__(self, path: str | Path, device: str = "cuda") -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.factory import get_policy_class

        self._torch = torch
        self.path = str(path)
        cfg = PreTrainedConfig.from_pretrained(self.path)
        slim = cfg.type == "pi05"
        cfg.device = "cpu" if slim else device  # o pi05 vai para cfg.device já no construtor
        policy = get_policy_class(cfg.type).from_pretrained(self.path, config=cfg)
        if slim:
            _slim_pi05(policy)
            cfg.device = device
        self.policy = policy.to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg, pretrained_path=self.path, preprocessor_overrides={"device_processor": {"device": device}}
        )
        self.name = f"{cfg.type}:{Path(self.path).name}"
        self.config = cfg

    def seed(self, seed: int) -> None:
        """Fixa o sorteio da inferência (o ruído do flow matching do pi0.5; o ACT não sorteia)."""
        self._torch.manual_seed(seed)

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def _batch(self, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
        torch = self._torch
        obs: dict[str, Any] = {
            "observation.state": torch.from_numpy(np.asarray(observation["agent_pos"], np.float32)),
            "task": instruction,
        }
        for cam, img in observation.get("pixels", {}).items():
            chw = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
            obs[f"observation.images.{cam}"] = chw.float() / 255.0
        return self.preprocessor(obs)

    def act(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        with self._torch.inference_mode():
            action = self.postprocessor(self.policy.select_action(self._batch(observation, instruction)))
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def predict_chunk(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        """Chunk inteiro previsto para esta observação, sem fila (teste offline). Shape (T, 6)."""
        with self._torch.inference_mode():
            batch = self._batch(observation, instruction)
            chunk = self.postprocessor(self.policy.predict_action_chunk(batch))
        return chunk.squeeze(0).cpu().numpy().astype(np.float32)
