"""Claims: 12 of 12 seeded model defects caught; 0 false rejects on 40
clean models.

Writes docs/quality_gate_output.txt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checks import run_all_checks
from src.touchstone_gen import generate_clean_models, generate_seeded_defects

ROOT = Path(__file__).resolve().parents[1]


def main():
    lines = []

    clean = generate_clean_models(n=40, seed=0)
    false_rejects = []
    for ntwk in clean:
        results = run_all_checks(ntwk)
        failed = [r.name for r in results if not r.passed]
        if failed:
            false_rejects.append((ntwk.name, failed))
    lines.append(f"Clean models: {len(clean)}. False rejects: {len(false_rejects)}.")
    for name, failed in false_rejects:
        lines.append(f"  FALSE REJECT {name}: {failed}")
    lines.append(f"Claim: '0 false rejects on 40 clean models'. Measured: "
                 f"{len(false_rejects)}/{len(clean)}. met: {len(false_rejects) == 0}")
    lines.append("")

    defects = generate_seeded_defects()
    n_caught = 0
    for name, ntwk in defects.items():
        results = run_all_checks(ntwk)
        failed = [r.name for r in results if not r.passed]
        caught = len(failed) > 0
        n_caught += caught
        lines.append(f"{name}: caught={caught} failed_checks={failed}")
    lines.append("")
    lines.append(f"Claim: '12 of 12 seeded defects caught'. Measured: {n_caught}/{len(defects)}. "
                 f"met: {n_caught == len(defects)}")

    out_path = ROOT / "docs" / "quality_gate_output.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
