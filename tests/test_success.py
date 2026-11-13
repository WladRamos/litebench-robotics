import numpy as np

from vials_lite.success import StepState, SuccessTracker, vial_geometry_ok

UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])
LYING = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # 90° em y
RACK_POS = np.array([0.18, 0.08, 0.06])
RACK_Q = UPRIGHT
IN_SLOT = RACK_POS + np.array([0.06, 0.06, 0.05])
FAR = np.array([0.30, -0.10, 0.05])


def state(step, pos0, q0, force0):
    pos = np.stack([pos0, FAR, FAR + [0, -0.08, 0]])
    quat = np.stack([q0, LYING, LYING])
    return StepState(step, pos, quat, RACK_POS, RACK_Q, np.array([force0, 0.0, 0.0]))


def test_geometry():
    assert vial_geometry_ok(IN_SLOT, UPRIGHT, RACK_POS, RACK_Q) == (True, True)
    assert vial_geometry_ok(IN_SLOT, LYING, RACK_POS, RACK_Q) == (False, True)
    assert vial_geometry_ok(FAR, UPRIGHT, RACK_POS, RACK_Q)[1] is False


def test_geometry_respects_rack_yaw():
    yaw = 0.5
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    local = np.array([0.06, 0.06, 0.05])
    c, s = np.cos(yaw), np.sin(yaw)
    world = RACK_POS + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1], local[2]])
    assert vial_geometry_ok(world, UPRIGHT, RACK_POS, q) == (True, True)


def run(seq):
    t = SuccessTracker()
    for k, (pos, q, f) in enumerate(seq):
        if t.update(state(k, pos, q, f)):
            return k
    return None


def test_success_needs_grasp_release_and_25_confirm_steps():
    seq = [(FAR, LYING, 0.0)] * 40  # aquecimento e aproximação
    seq += [(IN_SLOT, UPRIGHT, 5.0)] * 5  # agarrado, levado ao slot
    seq += [(IN_SLOT, UPRIGHT, 0.0)] * 40  # solto, parado no slot
    k = run(seq)
    assert k is not None
    first_release = 45
    assert k == first_release + 24  # 25 steps confirmados (contando o do disparo)


def test_preplaced_vial_without_grasp_is_not_success():
    assert run([(IN_SLOT, UPRIGHT, 0.0)] * 200) is None


def test_tilted_in_rack_is_not_success():
    seq = [(FAR, LYING, 0.0)] * 40 + [(IN_SLOT, LYING, 5.0)] * 5 + [(IN_SLOT, LYING, 0.0)] * 60
    assert run(seq) is None


def test_effective_grasp_window_depends_on_double_call_quirk():
    # Solta fora do rack e só depois de 15 steps o frasco aparece no slot.
    seq = [(FAR, LYING, 0.0)] * 40 + [(FAR, UPRIGHT, 5.0)] * 3 + [(FAR, UPRIGHT, 0.0)] * 15
    seq += [(IN_SLOT, UPRIGHT, 0.0)] * 60

    def run_with(writes):
        t = SuccessTracker(history_writes_per_step=writes)
        return any(t.update(state(k, *s)) for k, s in enumerate(seq))

    assert run_with(1) is True  # janela de 20 steps: ainda lembra do agarre
    assert run_with(2) is False  # janela efetiva de ~10 steps: esqueceu
