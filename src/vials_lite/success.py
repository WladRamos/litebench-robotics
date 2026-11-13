"""Critério de sucesso da tarefa, em NumPy puro e independente de simulador.

Porta de ``vial_placed_on_rack`` + ``vial_placed_on_rack_termination`` (mdp/terms.py do workshop).
Recebe, a cada step de controle (60 Hz), o estado privilegiado da cena e devolve se o
episódio terminou com sucesso. Qualquer simulador só precisa fornecer poses e a força de
contato garra↔frasco.

Regras (para cada frasco):
1. fora do aquecimento (``warmup_steps``);
2. aproximadamente vertical: |eixo z do frasco · z do mundo| > ``vertical_threshold``;
3. dentro da caixa do rack no frame local do rack (x, y em [0, 0.12]; z < 0.1);
4. foi agarrado (força > limiar) em algum momento da janela recente;
5. não está agarrado agora.
Quando algum frasco satisfaz 1–5 ele fica marcado como "colocado". O sucesso só é declarado
depois de ``confirm_steps`` (25) steps seguidos com algum frasco marcado ainda válido
(vertical, dentro da caixa, solto).

Peculiaridade reproduzida de propósito
--------------------------------------
No workshop, ``vial_placed_on_rack`` é chamada DUAS vezes por step: pelo termo de terminação e
pelo termo de observação ``subtask_terms.vial_placed``. O estado da função (buffer circular do
histórico de agarre) vive no objeto da função e é compartilhado, então o histórico avança duas
posições por step e a janela efetiva cai de 20 para ~10 steps. ``history_writes_per_step=2``
reproduz isso (padrão, para comparabilidade); use 1 para a semântica "pretendida".
Comportamento deduzido da leitura do código do workshop, não observado rodando o original.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .spec import SUCCESS, SuccessParams


def quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotaciona o vetor ``v`` pelo quaternion ``q`` (w, x, y, z)."""
    w, x, y, z = q_wxyz
    u = np.array([x, y, z])
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def quat_inverse(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    n = w * w + x * x + y * y + z * z
    return np.array([w, -x, -y, -z]) / n


def vial_geometry_ok(
    vial_pos: np.ndarray,
    vial_quat: np.ndarray,
    rack_pos: np.ndarray,
    rack_quat: np.ndarray,
    p: SuccessParams = SUCCESS,
) -> tuple[bool, bool]:
    """Retorna (vertical, dentro_da_caixa) para um frasco."""
    up = quat_rotate(np.asarray(vial_quat, float), np.array([0.0, 0.0, 1.0]))
    vertical = abs(up[2]) > p.vertical_threshold
    local = quat_rotate(quat_inverse(np.asarray(rack_quat, float)), np.asarray(vial_pos) - rack_pos)
    in_box = (
        p.rack_local_x[0] <= local[0] <= p.rack_local_x[1]
        and p.rack_local_y[0] <= local[1] <= p.rack_local_y[1]
        and local[2] < p.rack_local_z_max
    )
    return bool(vertical), bool(in_box)


@dataclass
class StepState:
    """Estado privilegiado num step. Arrays com uma linha por frasco."""

    step: int  # índice do step no episódio (0 no reset)
    vial_pos: np.ndarray  # (V, 3)
    vial_quat: np.ndarray  # (V, 4) wxyz
    rack_pos: np.ndarray  # (3,)
    rack_quat: np.ndarray  # (4,) wxyz
    jaw_contact_force: np.ndarray  # (V,) |força| garra↔frasco em N


@dataclass
class SuccessTracker:
    n_vials: int = 3
    params: SuccessParams = SUCCESS
    history_writes_per_step: int = 2  # ver docstring do módulo
    _history: np.ndarray = field(init=False)
    _hist_idx: int = field(init=False, default=0)
    _placed: np.ndarray = field(init=False)
    _confirm_active: bool = field(init=False, default=False)
    _counter: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._history = np.zeros((self.n_vials, self.params.grasp_history_window), dtype=bool)
        self._hist_idx = 0
        self._placed = np.zeros(self.n_vials, dtype=bool)
        self._confirm_active = False
        self._counter = 0

    @property
    def placed_flags(self) -> np.ndarray:
        return self._placed.copy()

    @property
    def confirm_counter(self) -> int:
        return self._counter

    def _placement_pass(self, s: StepState) -> np.ndarray:
        p = self.params
        in_warmup = s.step < p.warmup_steps
        grasped_now = s.jaw_contact_force > p.grasp_force_threshold_n
        self._history[:, self._hist_idx] = grasped_now
        recently = self._history.any(axis=1)
        for i in range(self.n_vials):
            vertical, in_box = vial_geometry_ok(s.vial_pos[i], s.vial_quat[i], s.rack_pos, s.rack_quat, p)
            if vertical and in_box and recently[i] and not grasped_now[i] and not in_warmup:
                self._placed[i] = True
        self._hist_idx = (self._hist_idx + 1) % p.grasp_history_window
        return self._placed.any() & (not in_warmup)

    def update(self, s: StepState) -> bool:
        """Processa um step; retorna True quando o sucesso é confirmado (terminated)."""
        if s.step <= 1:
            self.reset()
        trigger = False
        for _ in range(self.history_writes_per_step):
            trigger = self._placement_pass(s) or trigger
        if trigger and not self._confirm_active:
            self._confirm_active = True
            self._counter = 0
        still_valid = False
        if self._confirm_active:
            grasped_now = s.jaw_contact_force > self.params.grasp_force_threshold_n
            for i in np.flatnonzero(self._placed):
                vertical, in_box = vial_geometry_ok(
                    s.vial_pos[i], s.vial_quat[i], s.rack_pos, s.rack_quat, self.params
                )
                if vertical and in_box and not grasped_now[i]:
                    still_valid = True
        self._counter = self._counter + 1 if (self._confirm_active and still_valid) else 0
        if self._counter == 0:
            self._confirm_active = False
        return self._counter >= self.params.confirm_steps
