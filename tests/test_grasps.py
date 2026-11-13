"""Fechamentos da garra (evalkit.grasps) e a seção do relatório; NumPy puro."""

import numpy as np

from evalkit.grasps import (
    LIFT_WINDOW_STEPS,
    closure_stats,
    closures_from_arrays,
    find_closures,
    tracking_slope,
)
from evalkit.report import closures_section
from evalkit.taxonomy import Failure

LYING = np.array([np.sqrt(0.5), 0, np.sqrt(0.5), 0])  # eixo do frasco ao longo de +x
VIALS0 = np.array([[0.23, -0.08, 0.05], [0.23, 0.0, 0.05], [0.23, 0.08, 0.05]])


def episode(n=200):
    """Garra aberta; fecha no step 20 a 5 mm do eixo do frasco 0 (que sobe 5 cm) e no step 150 a
    5 cm do frasco 2 (que fica). O frasco 1 está mais perto do site no 1º fechamento, mas no rack."""
    cmd = np.full(n, 80.0)
    cmd[20:60] = 5.0
    cmd[150:] = 5.0
    site = np.tile([0.4, 0.3, 0.3], (n, 1))
    vial = np.tile(VIALS0, (n, 1, 1))
    site[20] = VIALS0[0] + [0.04, 0.005, 0.0]
    vial[20, 1] = site[20] + [0.0, 0.001, 0.0]  # frasco 1, "no rack", quase no site
    vial[40:, 0, 2] += 0.05
    site[150] = VIALS0[2] + [0.04, 0.05, 0.0]
    return {"action": np.column_stack([np.zeros((n, 5)), cmd]), "gripper_pos": site, "vial_pos": vial,
            "vial_quat": np.tile(LYING, (n, 3, 1))}


def test_closures_distance_lift_and_ignored_vial():
    a = episode()
    found = find_closures(a["action"][:, 5], a["gripper_pos"], a["vial_pos"], a["vial_quat"], ignore_vial=[1])
    assert [c.step for c in found] == [20, 150]
    first, second = found
    assert first.vial == 0 and first.axis_dist_mm == 5.0 and first.lifted
    assert first.vial_xy0 == (0.23, -0.08)
    assert second.vial == 2 and second.axis_dist_mm == 50.0 and not second.lifted
    # sem ignorar, o frasco 1 é o mais perto
    assert find_closures(a["action"][:, 5], a["gripper_pos"], a["vial_pos"], a["vial_quat"])[0].vial == 1


def test_lift_only_counts_inside_the_window():
    a = episode()
    a["vial_pos"][:, 0, 2] = VIALS0[0, 2]
    a["vial_pos"][20 + LIFT_WINDOW_STEPS + 5 :, 0, 2] += 0.05  # subiu, mas tarde demais
    rows = closures_from_arrays(a, ignore_vial=1)
    assert rows[0]["vial"] == 0 and not rows[0]["lifted"]


def test_tracking_slope():
    vial = np.linspace(-0.1, 0.0, 10)
    assert abs(tracking_slope(0.9 * vial + 0.02, vial) - 0.9) < 1e-9
    assert tracking_slope(np.zeros(10), np.zeros(10)) is None  # o frasco não varia
    assert tracking_slope(vial[:3], vial[:3]) is None  # poucos episódios


def test_closure_stats_and_report_section():
    never = Failure.NEVER_LIFTED.value

    def close(d, lifted, vy):
        return {"step": 30, "vial": 0, "axis_dist_mm": d, "lifted": lifted, "site_xy": (0.25, 0.5 * vy),
                "vial_xy0": (0.25 + vy, vy)}

    eps = [{"failure": "sucesso", "closures": [close(9.0, True, -0.01 * i)]} for i in range(5)]
    eps += [
        {"failure": never, "closures": [close(45.0, False, 0.0), close(60.0, False, 0.0)]},  # ao lado
        {"failure": never, "closures": [close(12.0, False, 0.01)]},  # sobre o frasco, escorregou
        {"failure": never, "closures": []},  # esbarrou sem fechar
        {"failure": "sucesso"},  # run antigo, sem a análise: fica de fora
    ]
    st = closure_stats(eps, never)
    assert st.n_episodes == 8 and st.n_closures == 8 and st.lifted == 5
    assert st.lifted_median_dist_mm == 9.0 and st.not_lifted_median_dist_mm == 45.0
    assert st.first_lifted == 5 and st.episodes_with_closure == 7
    assert abs(st.slope_y - 0.5) < 1e-9 and abs(st.slope_x) < 1e-9
    assert st.never_lifted_closed_beside == 2 and st.never_lifted_closed_on_vial == 1
    md = "\n".join(closures_section(eps))
    assert "**5/7**" in md and "y **0.50**" in md and "fechou só ao lado do frasco" in md
    assert closures_section([{"failure": "sucesso"}]) == []
