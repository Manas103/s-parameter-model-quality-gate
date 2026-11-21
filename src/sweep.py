"""Builds the 1,152-corner signoff sweep (capacitor tolerance x ESR
derating x regulator bandwidth, a full factorial design), runs the
droop model at every corner from a pinned manifest, and decomposes the
resulting droop variance by factor (a classical balanced-factorial ANOVA
sum-of-squares decomposition, computed directly from group means, since
the sweep has no repeats/noise to leave a residual error term).
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd

from src.droop_model import Corner, droop_v

CAP_TOLERANCE_LEVELS_PCT = [-20, -15, -10, -5, 0, 5, 10, 15, 20]  # 9 levels, standard part tolerance
# ESR derating over temperature and life is a large swing for real
# electrolytic/polymer capacitors (datasheets commonly show 2x-4x ESR
# increase from +25C to -40C, i.e. 100%-300% derating); the regulator's
# control-loop bandwidth, by contrast, is corner-screened as a narrower
# +-15% variation around a 200 kHz nominal design, since loop bandwidth
# is a design target the regulator is built to hold, not a part
# tolerance that drifts as widely as a capacitor's ESR does.
ESR_DERATING_LEVELS_PCT = [0, 40, 80, 120, 160, 200, 240, 280]  # 8 levels
REGULATOR_BANDWIDTH_LEVELS_HZ = list(np.linspace(170e3, 230e3, 16))  # 16 levels

N_CORNERS = len(CAP_TOLERANCE_LEVELS_PCT) * len(ESR_DERATING_LEVELS_PCT) * len(REGULATOR_BANDWIDTH_LEVELS_HZ)
assert N_CORNERS == 1152


def build_manifest() -> pd.DataFrame:
    """Enumerates every corner in a fixed (cap_tolerance, esr_derating,
    regulator_bandwidth) nested order and computes its droop. The order
    is deterministic (itertools.product over fixed, sorted level lists),
    which is what makes the manifest reproducible bit-for-bit."""
    rows = []
    for idx, (cap_pct, esr_pct, bw_hz) in enumerate(
        itertools.product(CAP_TOLERANCE_LEVELS_PCT, ESR_DERATING_LEVELS_PCT, REGULATOR_BANDWIDTH_LEVELS_HZ)
    ):
        corner = Corner(cap_pct, esr_pct, bw_hz)
        rows.append({
            "corner_index": idx,
            "cap_tolerance_pct": cap_pct,
            "esr_derating_pct": esr_pct,
            "regulator_bandwidth_hz": bw_hz,
            "droop_v": droop_v(corner),
        })
    return pd.DataFrame(rows)


def manifest_hash(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False, float_format="%.12g").encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def factorial_anova(df: pd.DataFrame) -> pd.DataFrame:
    """Classical balanced-factorial sum-of-squares decomposition of
    droop_v by the three factors and their interactions."""
    n_a = len(CAP_TOLERANCE_LEVELS_PCT)
    n_b = len(ESR_DERATING_LEVELS_PCT)
    n_c = len(REGULATOR_BANDWIDTH_LEVELS_HZ)
    y = df.sort_values(["cap_tolerance_pct", "esr_derating_pct", "regulator_bandwidth_hz"])["droop_v"].to_numpy()
    y = y.reshape(n_a, n_b, n_c)

    grand_mean = y.mean()
    mean_a = y.mean(axis=(1, 2))
    mean_b = y.mean(axis=(0, 2))
    mean_c = y.mean(axis=(0, 1))
    mean_ab = y.mean(axis=2)
    mean_ac = y.mean(axis=1)
    mean_bc = y.mean(axis=0)

    ss_total = float(np.sum((y - grand_mean) ** 2))
    ss_a = float(n_b * n_c * np.sum((mean_a - grand_mean) ** 2))
    ss_b = float(n_a * n_c * np.sum((mean_b - grand_mean) ** 2))
    ss_c = float(n_a * n_b * np.sum((mean_c - grand_mean) ** 2))

    eff_ab = mean_ab - mean_a[:, None] - mean_b[None, :] + grand_mean
    ss_ab = float(n_c * np.sum(eff_ab ** 2))
    eff_ac = mean_ac - mean_a[:, None] - mean_c[None, :] + grand_mean
    ss_ac = float(n_b * np.sum(eff_ac ** 2))
    eff_bc = mean_bc - mean_b[:, None] - mean_c[None, :] + grand_mean
    ss_bc = float(n_a * np.sum(eff_bc ** 2))

    ss_abc = ss_total - ss_a - ss_b - ss_c - ss_ab - ss_ac - ss_bc

    rows = [
        ("cap_tolerance", ss_a),
        ("esr_derating", ss_b),
        ("regulator_bandwidth", ss_c),
        ("cap_tolerance:esr_derating", ss_ab),
        ("cap_tolerance:regulator_bandwidth", ss_ac),
        ("esr_derating:regulator_bandwidth", ss_bc),
        ("three_way_interaction_and_nonlinearity", ss_abc),
    ]
    out = pd.DataFrame(rows, columns=["source", "sum_sq"])
    out["pct_of_total"] = out["sum_sq"] / ss_total * 100.0
    return out
