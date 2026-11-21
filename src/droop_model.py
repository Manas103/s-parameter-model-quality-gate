"""A deliberately simple, disclosed closed-form droop model for the
signoff sweep: not a from-scratch field or circuit solver (that is
project 1 and 2 in this set), but the kind of fast, closed-form estimate
a signoff *automation* tool runs across thousands of corners, trading
simulation fidelity for sweep volume. See README Honest framing.

droop(V) = I_step * ESR_effective + I_step * tau_reg / C_effective

The first term is the familiar instantaneous ESR drop; the second is a
standard simplified "depletion" estimate of how much extra the rail
sags before the regulator's control loop can respond (tau_reg), drawn
down through the effective decoupling capacitance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Corner:
    cap_tolerance_pct: float  # deviation from nominal capacitance, percent
    esr_derating_pct: float  # ESR increase from nominal, percent
    regulator_bandwidth_hz: float


NOMINAL_C_FARAD = 47e-6
NOMINAL_ESR_OHM = 4e-3
I_STEP_A = 8.0


def droop_v(corner: Corner) -> float:
    c_eff = NOMINAL_C_FARAD * (1.0 + corner.cap_tolerance_pct / 100.0)
    esr_eff = NOMINAL_ESR_OHM * (1.0 + corner.esr_derating_pct / 100.0)
    tau_reg = 1.0 / (2 * np.pi * corner.regulator_bandwidth_hz)
    return I_STEP_A * esr_eff + I_STEP_A * tau_reg / c_eff
