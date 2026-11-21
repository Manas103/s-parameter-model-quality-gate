"""Four independent screens for a multiport Touchstone S-parameter
model: passivity, reciprocity, a simplified time-domain causality
heuristic, and reference-impedance consistency.

Each check is implemented directly from the S-matrix and frequency/z0
arrays (not by calling scikit-rf's own `is_passive`/`is_reciprocal`
convenience properties); those are used in tests/test_checks.py as an
independent second implementation to cross-check against, the same role
ngspice plays for the other two repos in this set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skrf as rf


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_passivity(ntwk: rf.Network, tol: float = 1e-6) -> CheckResult:
    """A network is passive iff, at every frequency, the largest singular
    value of its S-matrix does not exceed 1 (it cannot deliver more
    power at any port combination than was incident)."""
    worst = 0.0
    worst_f = None
    for i, s in enumerate(ntwk.s):
        sigma_max = np.linalg.svd(s, compute_uv=False)[0]
        violation = sigma_max - 1.0
        if violation > worst:
            worst = violation
            worst_f = ntwk.f[i]
    passed = worst <= tol
    return CheckResult("passivity", passed,
                        f"max singular value exceeds 1 by {worst:.6g} at f={worst_f}")


def check_reciprocity(ntwk: rf.Network, tol: float = 1e-6) -> CheckResult:
    """A reciprocal network's S-matrix is symmetric at every frequency:
    S_ij = S_ji."""
    worst = 0.0
    worst_f = None
    for i, s in enumerate(ntwk.s):
        diff = np.max(np.abs(s - s.T))
        if diff > worst:
            worst = diff
            worst_f = ntwk.f[i]
    passed = worst <= tol
    return CheckResult("reciprocity", passed,
                        f"max |S_ij - S_ji| = {worst:.6g} at f={worst_f}")


def _acausal_energy_fraction(spectrum: np.ndarray) -> float:
    """IFFT-based causality screen: extend S(f), sampled uniformly from
    one step above DC to f_max, to a full real-signal spectrum (DC bin
    approximated by the lowest measured point, since there is no data
    exactly at DC; then the negative-frequency half as the complex
    conjugate mirror, the condition a real time-domain impulse response
    must satisfy), inverse-transform, and measure what fraction of the
    resulting impulse response's energy falls in the second half of the
    time record (where a response to an excitation at t=0 must be zero
    for a causal system)."""
    dc_bin = spectrum[0]
    full_spectrum = np.concatenate([[dc_bin], spectrum, np.conj(spectrum[::-1])])
    impulse = np.fft.ifft(full_spectrum)
    n = len(impulse)
    acausal_energy = np.sum(np.abs(impulse[n // 2:]) ** 2)
    total_energy = np.sum(np.abs(impulse) ** 2)
    return acausal_energy / total_energy if total_energy > 0 else 0.0


def check_causality(ntwk: rf.Network, tol: float = 0.35) -> CheckResult:
    """A simplified time-domain causality screen, not a full Kramers-
    Kronig (Hilbert-transform) relation: see _acausal_energy_fraction.
    A fully rigorous check would use the Hilbert-transform relation
    between the real and imaginary parts of S(f) over all frequency,
    including a reliable extrapolation to f=0 and f=infinity, which
    this screen does not attempt; see README Limitations. The
    tolerance is loose (0.35, i.e. up to about a third of the impulse
    response's energy in the "late" half) because truncating a
    physically causal network's spectrum to a finite measured band,
    with no data at DC or above f_max, itself produces some apparent
    energy there; egregious violations (phase reversal, randomized
    phase) produce fractions far above this band.
    """
    worst_frac = 0.0
    worst_pair = None
    n_port = ntwk.s.shape[1]
    for i in range(n_port):
        for j in range(n_port):
            frac = _acausal_energy_fraction(ntwk.s[:, i, j])
            if frac > worst_frac:
                worst_frac = frac
                worst_pair = (i, j)
    passed = worst_frac <= tol
    return CheckResult("causality", passed,
                        f"worst acausal energy fraction {worst_frac:.6g} at port pair {worst_pair} "
                        f"(tolerance {tol:.6g})")


def check_reference_impedance(ntwk: rf.Network, tol: float = 1e-6) -> CheckResult:
    """All ports must share one real, positive, frequency-independent
    reference impedance: no per-port mismatch, no per-frequency drift,
    and no stray reactive component in the declared z0."""
    z0 = ntwk.z0
    nominal = z0[0, 0]
    issues = []
    if abs(nominal.imag) > tol or nominal.real <= 0:
        issues.append(f"z0 not real-positive: {nominal}")
    max_dev = np.max(np.abs(z0 - nominal))
    if max_dev > tol:
        issues.append(f"z0 varies by up to {max_dev:.6g} across ports/frequency "
                       f"(nominal {nominal})")
    passed = len(issues) == 0
    return CheckResult("reference_impedance", passed, "; ".join(issues) if issues else "consistent")


def run_all_checks(ntwk: rf.Network) -> list[CheckResult]:
    return [
        check_passivity(ntwk),
        check_reciprocity(ntwk),
        check_causality(ntwk),
        check_reference_impedance(ntwk),
    ]
