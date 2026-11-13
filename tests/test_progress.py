"""Fases e resumo do episódio (evalkit.progress) e relatório; NumPy puro."""

import numpy as np

from evalkit.progress import RELEASE_DEBOUNCE_STEPS, Phase, ProgressState, ProgressTracker
from evalkit.report import funnel, render_report
from evalkit.taxonomy import Failure, classify
from vials_lite import spec

LYING = np.array([np.sqrt(0.5), 0, np.sqrt(0.5), 0])  # eixo do frasco ao longo de +x
UP = np.array([1.0, 0, 0, 0])
RACK_POS = np.array([0.18, 0.08, 0.041])
SLOT = RACK_POS + [0.03, 0.03, 0.014]  # frasco assentado no slot top_01


def state(gripper, vial, quat=LYING, force=0.0, success=False):
    other = np.array([[0.23, -0.16, 0.05], [0.23, 0.0, 0.05]])
    return ProgressState(
        gripper_pos=np.asarray(gripper, float),
        vial_pos=np.vstack([vial, other]),
        vial_quat=np.vstack([quat, LYING, LYING]),
        rack_pos=RACK_POS,
        rack_quat=UP,
        jaw_contact_force=np.array([force, 0.0, 0.0]),
        success=success,
    )


def test_phases_of_a_successful_pick_and_place():
    t = ProgressTracker()
    vial = np.array([0.23, -0.08, 0.05])
    assert t.update(state([0.4, 0.3, 0.3], vial)) is Phase.START
    # 5 cm além da origem, ao longo do eixo e 2 cm acima: alcançou
    assert t.update(state(vial + [0.05, 0, 0.02], vial)) is Phase.REACHED
    assert t.update(state(vial + [0.05, 0, 0.0], vial, force=5.0)) is Phase.GRASPED
    lifted = vial + [0, 0, 0.05]
    assert t.update(state(lifted + [0.05, 0, 0], lifted, force=5.0)) is Phase.LIFTED
    above = SLOT + [0, 0, 0.06]
    assert t.update(state(above, above, quat=UP, force=5.0)) is Phase.OVER_RACK
    # desce ainda agarrado (abaixo de 5,5 cm não é soltura) e solta no slot
    assert t.update(state(SLOT, SLOT, quat=UP, force=5.0)) is Phase.GRASPED
    for _ in range(RELEASE_DEBOUNCE_STEPS):
        phase = t.update(state(SLOT + [0, 0, 0.1], SLOT, quat=UP))
    assert phase is Phase.UPRIGHT
    assert t.update(state(SLOT + [0, 0, 0.1], SLOT, quat=UP, success=True)) is Phase.SUCCESS
    s = t.summary(success=True)
    assert s.n_grasps == 1 and s.n_releases_outside_rack == 0 and not s.released_inside_rack_tilted
    assert s.min_vial_to_slot_dist_m < 1e-3
    assert classify(s) is Failure.NONE


def test_regrasp_loop_counts_drops_outside_the_rack():
    t = ProgressTracker()
    vial = np.array([0.23, -0.08, 0.05])
    lifted = vial + [0, 0, 0.05]
    for _ in range(3):
        t.update(state(lifted, lifted, force=5.0))
        for _ in range(RELEASE_DEBOUNCE_STEPS):
            t.update(state(lifted + [0, 0, 0.05], vial))
    s = t.summary(success=False)
    assert s.n_grasps == 3 and s.n_releases_outside_rack == 3
    assert s.max_phase is Phase.LIFTED
    assert classify(s) is Failure.REGRASP_LOOP


def test_contact_flicker_is_not_a_release():
    t = ProgressTracker()
    lifted = np.array([0.23, -0.08, 0.1])
    t.update(state(lifted, lifted, force=5.0))
    t.update(state(lifted, lifted, force=0.0))  # 1 step sem contato
    t.update(state(lifted, lifted, force=5.0))
    assert t.n_grasps == 1 and t.n_releases_outside_rack == 0


def test_vial_already_in_rack_does_not_count_until_grasped():
    t = ProgressTracker(ignore_vial=0)
    assert t.update(state([0.4, 0.3, 0.3], SLOT, quat=UP)) is Phase.START
    assert t.update(state(SLOT + [0, 0, 0.05], SLOT, quat=UP)) is Phase.START


def test_tilted_release_inside_rack():
    t = ProgressTracker()
    above = SLOT + [0, 0, 0.06]
    t.update(state(above, above, quat=UP, force=5.0))
    for _ in range(RELEASE_DEBOUNCE_STEPS):
        phase = t.update(state(above + [0, 0, 0.1], SLOT, quat=LYING))
    assert phase is Phase.INSERTED
    s = t.summary(success=False)
    assert s.released_inside_rack_tilted and classify(s) is Failure.PLACED_TILTED


def test_report_funnel_and_table():
    eps = [
        {"episode": 0, "seed": 1, "success": True, "max_phase": "SUCCESS", "failure": "sucesso", "steps": 300,
         "n_grasps": 1, "n_releases_outside_rack": 0, "vial_on_rack_at_reset": None,
         "robot_color": "orange", "lighting": {"rect_exposure": 0.5}},
        {"episode": 1, "seed": 2, "success": False, "max_phase": "REACHED",
         "failure": Failure.NEVER_GRASPED.value, "steps": 450, "n_grasps": 0, "n_releases_outside_rack": 0,
         "vial_on_rack_at_reset": 2, "robot_color": "orange", "lighting": {"rect_exposure": -1.0}},
    ]
    f = dict(funnel(eps))
    assert f[Phase.REACHED] == 2 and f[Phase.GRASPED] == 1 and f[Phase.SUCCESS] == 1
    md = render_report({"policy_name": "act:x", "env": "vials_lite", "variant": "eval", "seed": 1,
                        "warmup_steps": spec.TIMING.warmup_steps_eval}, eps)
    assert "**1/2**" in md and Failure.NEVER_GRASPED.value in md


def test_vial_released_on_top_of_the_rack_missed_the_slot():
    t = ProgressTracker()
    on_top = RACK_POS + [0.06, 0.06, 0.09]  # em pé sobre o rack, entre os furos (z local < 0,1)
    t.update(state(on_top + [0, 0, 0.06], on_top, quat=UP, force=5.0))
    t.update(state(on_top + [0, 0, 0.06], on_top + [0, 0, 0.06], quat=UP, force=5.0))
    for _ in range(RELEASE_DEBOUNCE_STEPS):
        t.update(state(on_top + [0, 0, 0.2], on_top, quat=UP))
    assert t.summary(success=False).released_on_rack_top
    t.update(state(on_top + [0, 0, 0.2], on_top, quat=LYING))  # tombou: a causa continua sendo o furo
    s = t.summary(success=False)
    assert s.released_inside_rack_tilted and classify(s) is Failure.MISSED_SLOT


def test_several_vials_already_in_rack_are_ignored():
    t = ProgressTracker(ignore_vial=[0, 1])
    s = state(SLOT + [0, 0, 0.05], SLOT, quat=UP)
    s.vial_pos[1] = SLOT + [0.06, 0, 0]
    s.vial_quat[1] = UP
    s.gripper_pos = s.vial_pos[1] + [0, 0, 0.05]
    assert t.update(s) is Phase.START
