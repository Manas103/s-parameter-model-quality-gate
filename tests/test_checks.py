import numpy as np
import skrf as rf

from src.checks import check_causality, check_passivity, check_reciprocity, check_reference_impedance
from src.touchstone_gen import generate_clean_models, generate_seeded_defects, random_network, uniform_grid_from_near_dc


def test_clean_models_pass_every_check():
    for ntwk in generate_clean_models(n=40, seed=0):
        assert check_passivity(ntwk).passed, ntwk.name
        assert check_reciprocity(ntwk).passed, ntwk.name
        assert check_causality(ntwk).passed, ntwk.name
        assert check_reference_impedance(ntwk).passed, ntwk.name


def test_all_12_seeded_defects_are_caught_by_some_check():
    defects = generate_seeded_defects()
    assert len(defects) == 12
    for name, ntwk in defects.items():
        results = [check_passivity(ntwk), check_reciprocity(ntwk),
                   check_causality(ntwk), check_reference_impedance(ntwk)]
        assert any(not r.passed for r in results), f"{name} was not caught by any check"


def test_passivity_check_agrees_with_scikit_rf_is_passive():
    """Independent second implementation: scikit-rf's own is_passive
    property, computed from its own singular-value code path, used here
    only as a cross-check oracle, not as the gate's own implementation."""
    rng = np.random.default_rng(1)
    freq_hz = uniform_grid_from_near_dc(1e9, 32)
    clean = random_network(rng, 3, freq_hz)
    assert check_passivity(clean).passed
    assert bool(np.all(clean.passivity <= 1 + 1e-9))

    defects = generate_seeded_defects()
    bad = defects["passivity_1_single_point_gain"]
    assert not check_passivity(bad).passed
    assert not bool(np.all(bad.passivity <= 1 + 1e-9))


def test_reciprocity_check_agrees_with_scikit_rf_is_reciprocal():
    rng = np.random.default_rng(2)
    freq_hz = uniform_grid_from_near_dc(1e9, 32)
    clean = random_network(rng, 3, freq_hz)
    assert check_reciprocity(clean).passed
    assert clean.is_reciprocal()

    defects = generate_seeded_defects()
    bad = defects["reciprocity_1_port12_asymmetry"]
    assert not check_reciprocity(bad).passed
    assert not bad.is_reciprocal()


def test_causality_time_reversal_is_detected():
    defects = generate_seeded_defects()
    assert not check_causality(defects["causality_1_full_time_reversal"]).passed


def test_reference_impedance_defects_are_detected():
    defects = generate_seeded_defects()
    for name in ("z0_1_port_mismatch", "z0_2_frequency_drift", "z0_3_reactive"):
        assert not check_reference_impedance(defects[name]).passed, name
