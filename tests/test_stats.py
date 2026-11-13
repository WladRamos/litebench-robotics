import pytest

from evalkit.stats import mcnemar_exact, newcombe_difference, paired_difference, wilson_interval


def test_wilson_reference_value():
    ci = wilson_interval(11, 20)
    assert ci.estimate == pytest.approx(0.55)
    assert ci.low == pytest.approx(0.342, abs=1e-3)
    assert ci.high == pytest.approx(0.742, abs=1e-3)


def test_wilson_edges():
    assert wilson_interval(0, 10).low == pytest.approx(0.0)
    assert wilson_interval(10, 10).high == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wilson_interval(3, 0)


def test_newcombe_20_episodes_cannot_separate_55_from_70():
    d = newcombe_difference(14, 20, 11, 20)
    assert d.low < 0 < d.high


def test_newcombe_large_gap_is_significant():
    d = newcombe_difference(90, 100, 50, 100)
    assert d.low > 0


def test_paired_reference_value():
    # Exemplo da MetricGate ("Score confidence interval for the difference of paired proportions"):
    # 30 pares, 8 e 8 discordantes; o "Newcombe square-and-add" dá 0 [−0,2462; 0,2462].
    d = paired_difference(8, 8, 8, 6)
    assert d.estimate == pytest.approx(0.0)
    assert d.low == pytest.approx(-0.2462, abs=1e-4)
    assert d.high == pytest.approx(0.2462, abs=1e-4)


def test_paired_is_narrower_than_unpaired_when_pairs_agree():
    # 100 pares com 80% de concordância: o pareado aproveita a correlação.
    p = paired_difference(15, 5, 10, 70)
    u = newcombe_difference(20, 100, 25, 100)
    assert p.estimate == pytest.approx(u.estimate)
    assert p.high - p.low < u.high - u.low
    assert p.low < 0 < p.high


def test_paired_edges():
    assert paired_difference(0, 0, 0, 10).estimate == 0
    d = paired_difference(0, 10, 0, 0)
    assert d.estimate == pytest.approx(1.0) and d.high <= 1.0 and d.low > 0
    with pytest.raises(ValueError):
        paired_difference(0, 0, 0, 0)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(11, 11) == 1.0
    assert mcnemar_exact(24, 6) == pytest.approx(0.0014, abs=1e-4)
    assert mcnemar_exact(21, 7) == pytest.approx(0.0125, abs=1e-3)  # p = 0,013
