"""Generates synthetic Touchstone models: physically-valid clean models
(guaranteed passive, reciprocal, and causal by construction, since they
come from real R/L/C component values evaluated at real frequencies)
and 12 seeded defects, each built to trip exactly one of the four checks
in src/checks.py.
"""
from __future__ import annotations

import numpy as np
import skrf as rf


def _random_branch_z(rng: np.random.Generator, freq_hz: np.ndarray) -> np.ndarray:
    """A random series R(+L)(+1/C) branch impedance; always positive-real
    resistance plus a reactive term, so it is passive by construction.

    L and C are sized so each reactive corner frequency (where its
    reactance equals R) falls inside the model's own measured band, not
    far below its lowest sampled point. A corner far below the band
    means the branch's low-frequency behavior is fully determined by
    physics the sampled band never observes, which is a real cause of
    the IFFT-based causality screen's truncated-band artifacts (see
    src/checks.py); tying the corner to the band avoids that regardless
    of which frequency range a given model happens to use.
    """
    w = 2 * np.pi * freq_hz
    r = rng.uniform(0.5, 50.0)
    z = np.full(len(freq_hz), r, dtype=complex)
    f_lo, f_hi = freq_hz[0] * 3, freq_hz[-1] / 3
    if rng.random() < 0.7:
        f_c = np.exp(rng.uniform(np.log(f_lo), np.log(f_hi)))
        l_henry = r / (2 * np.pi * f_c)
        z += 1j * w * l_henry
    if rng.random() < 0.7:
        f_c = np.exp(rng.uniform(np.log(f_lo), np.log(f_hi)))
        c_farad = 1.0 / (2 * np.pi * f_c * r)
        z += 1.0 / (1j * w * c_farad)
    return z


def random_network(rng: np.random.Generator, n_ports: int, freq_hz: np.ndarray,
                    z0: float = 50.0) -> rf.Network:
    """A star topology (as in the PDN optimizer and IR-drop repos in
    this set): a shared rail branch plus one branch per port, all built
    from random passive R/L/C elements, which makes the resulting
    network passive and reciprocal by construction (it is literally the
    impedance matrix of a physical network of positive resistances,
    inductances and capacitances)."""
    rail = _random_branch_z(rng, freq_hz)
    branches = [_random_branch_z(rng, freq_hz) for _ in range(n_ports)]

    n_freq = len(freq_hz)
    z = np.zeros((n_freq, n_ports, n_ports), dtype=complex)
    for i in range(n_ports):
        for j in range(n_ports):
            z[:, i, j] = rail
    for k in range(n_ports):
        z[:, k, k] += branches[k]

    freq = rf.Frequency.from_f(freq_hz, unit="hz")
    z0_arr = np.full((n_freq, n_ports), z0, dtype=complex)
    ntwk = rf.Network(frequency=freq, z0=z0_arr)
    ntwk.z = z
    return ntwk


def uniform_grid_from_near_dc(f_stop_hz: float, n_points: int) -> np.ndarray:
    """A uniform frequency grid from one frequency step above DC to
    f_stop_hz. src.checks.check_causality's IFFT-based heuristic treats
    the sampled band as starting at (near) DC, so every generated model
    (clean or seeded-defect) uses this grid shape, not an arbitrary
    [f_start, f_stop] band that would make that heuristic meaningless."""
    f_step = f_stop_hz / n_points
    return np.linspace(f_step, f_stop_hz, n_points)


def generate_clean_models(n: int = 40, seed: int = 0) -> list[rf.Network]:
    rng = np.random.default_rng(seed)
    models = []
    for i in range(n):
        n_ports = int(rng.integers(2, 6))
        n_points = int(rng.integers(21, 101))
        f_stop = rng.uniform(1e8, 5e10)
        freq_hz = uniform_grid_from_near_dc(f_stop, n_points)
        ntwk = random_network(rng, n_ports, freq_hz)
        ntwk.name = f"clean_{i:03d}"
        models.append(ntwk)
    return models


def _baseline(seed: int = 12345) -> rf.Network:
    rng = np.random.default_rng(seed)
    freq_hz = uniform_grid_from_near_dc(1e9, 64)
    return random_network(rng, 3, freq_hz)


def generate_seeded_defects(rng_seed: int = 7) -> dict[str, rf.Network]:
    """12 defects, 3 independently-constructed variants for each of the
    four checks in src/checks.py, all derived from the same clean
    baseline network so the checks are what catches them, not an
    accidentally-malformed baseline."""
    defects: dict[str, rf.Network] = {}
    base = _baseline()
    n_freq, n_port, _ = base.s.shape
    rng = np.random.default_rng(rng_seed)

    # --- passivity (3 variants) ---
    d = base.copy()
    d.s[n_freq // 2, 0, 0] = 1.5 + 0.0j
    defects["passivity_1_single_point_gain"] = d

    d = base.copy()
    phase = np.angle(d.s[n_freq // 2:, 1, 1])
    d.s[n_freq // 2:, 1, 1] = 1.3 * np.exp(1j * phase)
    defects["passivity_2_band_gain"] = d

    d = base.copy()
    d.s[:, 2, 2] = 1.1 + 0.0j
    defects["passivity_3_reflection_above_unity"] = d

    # --- reciprocity (3 variants) ---
    d = base.copy()
    d.s[:, 0, 1] = d.s[:, 0, 1] * 1.3
    defects["reciprocity_1_port12_asymmetry"] = d

    d = base.copy()
    d.s[:, 0, 2] = d.s[:, 0, 2] + 0.1
    defects["reciprocity_2_port13_offset"] = d

    d = base.copy()
    d.s[:, 1, 2], d.s[:, 2, 1] = d.s[:, 2, 1].copy(), d.s[:, 1, 2].copy()
    d.s[:, 1, 2] *= 1.5
    defects["reciprocity_3_port23_swap_and_scale"] = d

    # --- causality (3 variants) ---
    d = base.copy()
    d.s = np.conj(d.s)
    defects["causality_1_full_time_reversal"] = d

    d = base.copy()
    d.s[:, 0, 1] = np.conj(d.s[:, 0, 1])
    defects["causality_2_single_pair_time_reversal"] = d

    d = base.copy()
    mags = np.abs(d.s)
    random_phase = rng.uniform(-np.pi, np.pi, size=d.s.shape)
    d.s = mags * np.exp(1j * random_phase)
    defects["causality_3_scrambled_phase"] = d

    # --- reference impedance (3 variants) ---
    d = base.copy()
    z0 = d.z0.copy()
    z0[:, 1] = 75.0
    d.z0 = z0
    defects["z0_1_port_mismatch"] = d

    d = base.copy()
    z0 = d.z0.copy()
    z0[:, 0] = z0[:, 0] + np.linspace(0, 5.0, n_freq)
    d.z0 = z0
    defects["z0_2_frequency_drift"] = d

    d = base.copy()
    z0 = d.z0.copy()
    z0[:, :] = 50.0 + 10.0j
    d.z0 = z0
    defects["z0_3_reactive"] = d

    assert len(defects) == 12
    return defects
