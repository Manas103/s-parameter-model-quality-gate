import numpy as np

from src.droop_model import Corner, droop_v
from src.sweep import N_CORNERS, build_manifest, factorial_anova, manifest_hash


def test_manifest_has_1152_corners():
    df = build_manifest()
    assert len(df) == 1152 == N_CORNERS
    assert df["corner_index"].nunique() == 1152


def test_manifest_is_bit_for_bit_reproducible():
    df1 = build_manifest()
    df2 = build_manifest()
    assert manifest_hash(df1) == manifest_hash(df2)


def test_manifest_corner_order_is_deterministic_itertools_product():
    df = build_manifest()
    assert df.iloc[0]["cap_tolerance_pct"] == -20
    assert df.iloc[0]["esr_derating_pct"] == 0
    assert df.iloc[-1]["cap_tolerance_pct"] == 20
    assert df.iloc[-1]["esr_derating_pct"] == 280


def test_droop_increases_with_esr_derating_all_else_fixed():
    base = Corner(cap_tolerance_pct=0, esr_derating_pct=0, regulator_bandwidth_hz=200e3)
    derated = Corner(cap_tolerance_pct=0, esr_derating_pct=100, regulator_bandwidth_hz=200e3)
    assert droop_v(derated) > droop_v(base)


def test_droop_increases_as_regulator_bandwidth_decreases():
    slow = Corner(cap_tolerance_pct=0, esr_derating_pct=0, regulator_bandwidth_hz=170e3)
    fast = Corner(cap_tolerance_pct=0, esr_derating_pct=0, regulator_bandwidth_hz=230e3)
    assert droop_v(slow) > droop_v(fast)


def test_anova_sum_of_squares_adds_up_to_total():
    """Sanity check on the factorial decomposition itself: for a
    balanced full factorial with no repeats, every source's sum of
    squares (including interactions) must add up to exactly the total
    sum of squares, with no residual left over."""
    df = build_manifest()
    anova = factorial_anova(df)
    y = df["droop_v"].to_numpy()
    ss_total = float(np.sum((y - y.mean()) ** 2))
    assert np.isclose(anova["sum_sq"].sum(), ss_total, rtol=1e-9)
    assert np.isclose(anova["pct_of_total"].sum(), 100.0, atol=1e-6)


def test_esr_derating_is_the_dominant_variance_source():
    df = build_manifest()
    anova = factorial_anova(df)
    esr_row = anova[anova["source"] == "esr_derating"].iloc[0]
    others = anova[anova["source"] != "esr_derating"]["pct_of_total"]
    assert esr_row["pct_of_total"] > others.max()
