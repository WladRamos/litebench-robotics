"""Vials-Lite: a tarefa Vials-to-Rack do workshop Sim-to-Real SO-101 da NVIDIA, em MuJoCo.

Módulos:
    spec           — constantes da tarefa extraídas do workshop (fonte única de verdade)
    joint_mapping  — LeRobot ↔ radianos (porta do LeRobotSO101Interface)
    success        — critério de sucesso em NumPy, independente de simulador
    scene          — cena MuJoCo (MJCF do Menagerie + frascos, rack, tapete, lightbox, luz, câmeras)
    env            — ambiente Gymnasium ``VialsLiteEnv`` (variantes, arranjo, cenas fixas)
    scripted       — demo roteirizada com estado privilegiado
    lerobot_env    — plugin do ``lerobot-eval`` e ``make_env`` do EnvHub (precisa do extra lerobot)
"""

__version__ = "0.0.1"
