"""Ponto de entrada do EnvHub do LeRobot para o Vials-Lite.

O ``lerobot-eval`` baixa este arquivo do repositório do Hub e chama ``make_env(n_envs,
use_async_envs[, cfg])``. A implementação fica no pacote ``vials_lite``, que precisa estar instalado
na máquina que avalia (``pip install "litebench-robotics[sim,lerobot]"``; ver ``envhub/README.md``).
"""

from vials_lite.lerobot_env import make_env

__all__ = ["make_env"]
