from evalkit.progress import Phase, phase_progress
from evalkit.taxonomy import EpisodeSummary, Failure, classify


def ep(**kw):
    base = dict(success=False, max_phase=Phase.START, n_grasps=0, n_releases_outside_rack=0,
                released_inside_rack_tilted=False, min_vial_to_slot_dist_m=1.0)
    base.update(kw)
    return EpisodeSummary(**base)


def test_progress_bounds():
    assert phase_progress(Phase.START) == 0.0
    assert phase_progress(Phase.SUCCESS) == 1.0


def test_classification():
    assert classify(ep(success=True, max_phase=Phase.SUCCESS)) is Failure.NONE
    assert classify(ep()) is Failure.NEVER_REACHED
    assert classify(ep(max_phase=Phase.REACHED)) is Failure.NEVER_GRASPED
    assert classify(ep(max_phase=Phase.OVER_RACK, n_grasps=4)) is Failure.REGRASP_LOOP
    tilted = ep(max_phase=Phase.INSERTED, n_grasps=1, released_inside_rack_tilted=True)
    assert classify(tilted) is Failure.PLACED_TILTED
    dropped = ep(max_phase=Phase.LIFTED, n_grasps=1, n_releases_outside_rack=1)
    assert classify(dropped) is Failure.DROPPED_IN_TRANSIT


def test_touched_but_never_lifted():
    assert classify(ep(max_phase=Phase.GRASPED)) is Failure.NEVER_LIFTED
