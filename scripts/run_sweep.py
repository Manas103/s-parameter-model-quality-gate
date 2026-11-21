"""Claims: a 1,152-corner sweep over capacitor tolerance, ESR derating
and regulator response; it reruns bit-for-bit from a pinned manifest;
and a factorial analysis attributes 62% of worst-case droop variance to
ESR derating.

Writes docs/sweep_output.txt and docs/sweep_manifest.csv (the pinned
manifest).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sweep import N_CORNERS, build_manifest, factorial_anova, manifest_hash

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = build_manifest()
    h1 = manifest_hash(df)

    manifest_path = ROOT / "docs" / "sweep_manifest.csv"
    df.to_csv(manifest_path, index=False, float_format="%.12g")

    df_rerun = build_manifest()
    h2 = manifest_hash(df_rerun)
    bit_for_bit = h1 == h2

    anova = factorial_anova(df)
    esr_pct = float(anova.loc[anova["source"] == "esr_derating", "pct_of_total"].iloc[0])

    lines = []
    lines.append(f"Claim: '1,152-corner sweep over capacitor tolerance, ESR derating and "
                 f"regulator response'. Measured: {len(df)} corners "
                 f"(9 x 8 x 16). met: {len(df) == N_CORNERS == 1152}")
    lines.append("")
    lines.append(f"Claim: 'reruns bit-for-bit from a pinned manifest'. Manifest sha256 "
                 f"(run 1): {h1}")
    lines.append(f"Manifest sha256 (run 2, independent rebuild): {h2}")
    lines.append(f"Bit-for-bit match: {bit_for_bit}. met: {bit_for_bit}")
    lines.append("")
    lines.append("Factorial ANOVA, droop variance across all 1,152 corners:")
    lines.append(anova.to_string(index=False))
    lines.append("")
    lines.append(f"Worst-case (max) droop across the sweep: {df['droop_v'].max():.6f} V")
    lines.append(f"Claim: 'factorial analysis attributes 62% of worst-case droop variance to "
                 f"ESR derating'. Measured: {esr_pct:.2f}%. met (within 2 points): "
                 f"{abs(esr_pct - 62.0) < 2.0}")

    out_path = ROOT / "docs" / "sweep_output.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
