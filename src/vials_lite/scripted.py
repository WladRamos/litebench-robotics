"""Política roteirizada com estado privilegiado: baseline de referência e sanidade do evalkit.

Faz um pick-and-place completo: escolhe um frasco deitado e um slot livre, agarra como os operadores
do dataset (aproximação 59° abaixo da horizontal no plano do braço, fechamento horizontal, folga da
mandíbula fixa), levanta, põe o frasco em pé (tampa para cima, como o operador faz
com o wrist_flex) sobre o slot, desce, solta e recua.

A IK roda numa cópia do ``MjData`` do ambiente (mínimos quadrados amortecidos nas 5 juntas do braço).
O plano é feito uma vez por fase, com o estado real do momento; as ações vão para o ambiente no espaço
do LeRobot, interpoladas em ``n`` steps de 60 Hz.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from . import spec
from .env import VialsLiteEnv, _rotate
from .joint_mapping import mjcf_radians_to_usd, radians_to_lerobot

APPROACH_PITCH_DEG = 59.0
JAW_CLEARANCE = 0.012
GRASP_DEPTH = 0.012
FIXED_JAW_Z = -0.020  # mandíbula fixa no frame do gripperframe (z = eixo de fechamento)
GRIPPER_OPEN = 0.9  # rad (MJCF)
GRIPPER_CLOSED = -0.17  # rad: limite inferior; o frasco trava a garra antes
UP = np.array([0.0, 0.0, 1.0])
# Base do frasco (origem) no frame do rack ao soltar; topo do rack em 0,073. Mais fundo que isso o
# braço chega aos limites (elbow +97°, wrist_flex −95°); o frasco termina de descer sozinho.
INSERT_LOCAL_Z = 0.07
# Semente da IK para pôr o frasco em pé: postura mediana dos operadores ao soltar (75 episódios,
# graus no MJCF: pan, lift, elbow, wrist_flex, wrist_roll).
OPERATOR_RELEASE_DEG = (-28.0, -8.0, 69.0, -68.0, -89.0)


class ScriptedPickPlace:
    """Gera ações (LeRobot, (6,)) para um episódio; use ``run`` ou itere ``actions``."""

    def __init__(self, env: VialsLiteEnv, vial: int | None = None, slot: int | None = None) -> None:
        import mujoco

        self._mj = mujoco
        self.env = env
        self.m = env.model
        self.ik_data = mujoco.MjData(env.model)
        self.vial = vial
        self.slot = slot
        self.ik_errors: dict[str, float] = {}

    # ------------------------------------------------------------------ IK
    def _ik(
        self,
        q0: np.ndarray,
        point_local: np.ndarray,
        target: np.ndarray,
        dirs: list[tuple[np.ndarray, np.ndarray, float]],
        iters: int = 400,
    ) -> tuple[np.ndarray, float]:
        """Leva ``site + R·point_local`` a ``target``; cada (v_local, v_mundo, peso) alinha R·v_local."""
        mj, m, d, ids = self._mj, self.m, self.ik_data, self.env.ids
        arm = ids.arm_qpos[:5]
        dof = ids.arm_qvel[:5]
        d.qpos[:] = self.env.data.qpos
        d.qpos[arm] = q0
        lo, hi = m.jnt_range[[m.joint(j).id for j in spec.JOINTS_MENAGERIE[:5]]].T
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        body = m.site_bodyid[ids.gripper_site]
        err = np.inf
        for _ in range(iters):
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            R = d.site_xmat[ids.gripper_site].reshape(3, 3)
            p = d.site_xpos[ids.gripper_site] + R @ point_local
            e_p = target - p
            err = float(np.linalg.norm(e_p))
            mj.mj_jac(m, d, jacp, jacr, p, body)
            rows, errs = [jacp[:, dof]], [e_p]
            for v_local, v_world, w in dirs:
                v = R @ v_local
                errs.append(w * np.cross(v, v_world))
                rows.append(w * jacr[:, dof])
            J, e = np.vstack(rows), np.concatenate(errs)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
            if np.linalg.norm(dq) < 1e-7:
                break
            d.qpos[arm] = np.clip(d.qpos[arm] + dq, lo, hi)
        return d.qpos[arm].copy(), err

    # ------------------------------------------------------------------ escolhas
    def _choose(self) -> tuple[int, int]:
        p = self.env._info(False, np.zeros(3))["privileged"]
        rack_pos, rack_quat = p["rack_pos"], p["rack_quat"]
        slots = [rack_pos + _rotate(rack_quat, np.array(s)) for s in spec.RACK_SLOTS_LOCAL]
        on_rack = p["vial_on_rack_at_reset"]
        vial = self.vial
        if vial is None:
            vial = next(i for i in range(len(spec.VIALS)) if i != on_rack)
        slot = self.slot
        if slot is None:
            occupied = (
                {int(np.argmin([np.linalg.norm(p["vial_pos"][on_rack, :2] - s[:2]) for s in slots]))}
                if on_rack is not None
                else set()
            )
            base = np.array(spec.MJCF_BASE_POS)
            free = [k for k in range(len(slots)) if k not in occupied]
            slot = min(free, key=lambda k: np.linalg.norm(slots[k][:2] - base[:2]))  # o mais perto da base
        return vial, slot

    # ------------------------------------------------------------------ roteiro
    def _move(self, q_arm: np.ndarray, grip: float, n: int) -> Iterator[np.ndarray]:
        """Interpola do último comando até (q_arm, grip) em n steps; rende ações LeRobot."""
        q_to = np.append(q_arm, grip)
        q_from = self._last_cmd
        for k in range(n):
            a = (k + 1) / n
            s = a * a * (3 - 2 * a)
            q = q_from + s * (q_to - q_from)
            self._last_cmd = q
            yield radians_to_lerobot(mjcf_radians_to_usd(q))

    def actions(self) -> Iterator[np.ndarray]:
        env, ids = self.env, self.env.ids
        self._last_cmd = env.data.ctrl[ids.actuators].copy()
        q_now = env.data.qpos[ids.arm_qpos[:5]].copy()
        vial, slot = self._choose()
        self.choice = (spec.VIALS[vial].name, slot)

        # Espera o aquecimento do critério (30 steps), com a garra abrindo.
        yield from self._move(q_now, GRIPPER_OPEN, spec.SUCCESS.warmup_steps)

        # --- agarre (como os operadores)
        vb = ids.vial_bodies[vial]
        R_v = env.data.xmat[vb].reshape(3, 3)
        axis = R_v[:, 2]
        center = env.data.xpos[vb] + axis * (spec.VIAL_BODY_TOP_Z / 2)
        horiz = center - env.data.xpos[self.m.body("shoulder").id]
        horiz[2] = 0.0
        horiz /= np.linalg.norm(horiz)
        pitch = np.deg2rad(APPROACH_PITCH_DEG)
        x_pref = np.cos(pitch) * horiz - np.sin(pitch) * UP
        z_des = np.cross(x_pref, UP)
        z_des /= np.linalg.norm(z_des)
        grasp_local = np.array([-GRASP_DEPTH, 0.0, FIXED_JAW_Z + spec.VIAL_BODY_RADIUS + JAW_CLEARANCE])
        x_local, z_local = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
        best = None
        for sign in (1, -1):
            dirs = [(z_local, sign * z_des, 0.3), (x_local, x_pref, 0.3)]
            q_g, e = self._ik(q_now, grasp_local, center, dirs)
            roll = q_g[4]
            if best is None or (e < 0.005, -abs(roll)) > (best[1] < 0.005, -abs(best[0][4])):
                best = (q_g, e, dirs)
        q_grasp, e_grasp, dirs = best
        q_pre, e_pre = self._ik(q_grasp, grasp_local, center + 0.05 * UP, dirs)
        self.ik_errors.update(grasp=e_grasp, pre=e_pre)
        yield from self._move(q_pre, GRIPPER_OPEN, 50)
        yield from self._move(q_grasp, GRIPPER_OPEN, 35)
        yield from self._move(q_grasp, GRIPPER_CLOSED, 30)
        yield from self._move(q_grasp, GRIPPER_CLOSED, 10)

        # --- levanta 8 cm por continuação
        q = q_grasp
        for h in np.arange(0.01, 0.081, 0.01):
            q, e_lift = self._ik(q, grasp_local, center + h * UP, dirs)
        self.ik_errors["lift"] = e_lift
        yield from self._move(q, GRIPPER_CLOSED, 40)

        # --- frasco no frame do gripperframe, medido agora (depois do agarre real)
        d = env.data
        R_s = d.site_xmat[ids.gripper_site].reshape(3, 3)
        p_s = d.site_xpos[ids.gripper_site]
        vial_origin_local = R_s.T @ (d.xpos[vb] - p_s)
        vial_axis_local = R_s.T @ d.xmat[vb].reshape(3, 3)[:, 2]

        # --- em pé sobre o slot: origem do frasco acima da entrada, eixo para cima
        info = env._info(False, np.zeros(3))["privileged"]
        slot_w = info["rack_pos"] + _rotate(info["rack_quat"], np.array(spec.RACK_SLOTS_LOCAL[slot]))
        upright = [(vial_axis_local, UP, 1.0)]  # posição (3) + eixo (2) = 5 juntas
        above = slot_w + np.array([0, 0, 0.04])
        q_mid, _ = self._ik(
            np.deg2rad(OPERATOR_RELEASE_DEG), vial_origin_local, (center + 0.08 * UP + above) / 2, upright
        )
        q_above, e_above = self._ik(q_mid, vial_origin_local, above, upright)
        self.ik_errors["above"] = e_above
        yield from self._move(q_mid, GRIPPER_CLOSED, 40)
        yield from self._move(q_above, GRIPPER_CLOSED, 40)
        yield from self._move(q_above, GRIPPER_CLOSED, 20)  # assenta

        # --- desce por continuação (1 cm por vez) até INSERT_LOCAL_Z; ao soltar, o frasco cai no furo
        inside = info["rack_pos"][2] + INSERT_LOCAL_Z
        q_in = q_above
        for z in np.arange(above[2] - 0.01, inside - 1e-9, -0.01):
            q_in, e_in = self._ik(q_in, vial_origin_local, np.array([slot_w[0], slot_w[1], z]), upright)
        q_in, e_in = self._ik(q_in, vial_origin_local, np.array([slot_w[0], slot_w[1], inside]), upright)
        self.ik_errors["insert"] = e_in
        yield from self._move(q_in, GRIPPER_CLOSED, 40)

        # --- solta e recua para cima
        yield from self._move(q_in, GRIPPER_OPEN, 20)
        q_out, _ = self._ik(q_in, vial_origin_local, np.array([slot_w[0], slot_w[1], inside + 0.10]), upright)
        yield from self._move(q_out, GRIPPER_OPEN, 40)
        while True:  # segura até o fim do episódio
            yield from self._move(q_out, GRIPPER_OPEN, 1)

    def run(self, seed: int | None = None, max_steps: int = spec.TIMING.episode_steps) -> dict:
        """Reset + episódio inteiro; devolve resumo (sucesso, steps, eventos, erros de IK)."""
        self.env.reset(seed=seed)
        events: list[tuple[int, str]] = []
        terminated = truncated = False
        info: dict = {}
        it = self.actions()
        while not (terminated or truncated):
            _, _, terminated, truncated, info = self.env.step(next(it))
            events += [(info["privileged"]["step"], e) for e in info["privileged"]["events"]]
            if info["privileged"]["step"] >= max_steps:
                break
        return {
            "success": bool(info.get("is_success")),
            "steps": info["privileged"]["step"],
            "events": events,
            "choice": self.choice,
            "ik_errors_mm": {k: round(1000 * v, 1) for k, v in self.ik_errors.items()},
        }
