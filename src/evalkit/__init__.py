"""evalkit: avaliação diagnóstica e estatística de políticas de manipulação.

    stats     — IC de Wilson, diferença de proporções (Newcombe), McNemar exato
    progress  — fases e progresso a partir de estado privilegiado
    taxonomy  — classificação automática de falhas
    grasps    — fechamentos da garra: onde fecha em relação ao frasco, se levanta, se segue o frasco
    policy    — protocolos Policy / Probe
    runner    — N episódios de uma política no Vials-Lite, com estado por step (``python -m evalkit.runner``)
    lerobot_policy — backend LeRobot (from_pretrained + select_action)
    report    — relatório Markdown de um run
    groot_bridge — protocolo do servidor GR00T: serve políticas LeRobot ao avaliador do workshop no Isaac
"""

__version__ = "0.0.1"
