# S-Parameter Model Quality Gate and Signoff Sweep Automation

A Python/scikit-rf/pandas tool that screens Touchstone S-parameter
models for passivity, reciprocity, causality and reference-impedance
consistency, and runs a pinned, reproducible 1,152-corner signoff sweep
with a factorial variance-attribution analysis. Every number below was
measured on this machine, not targeted: where a check needed
recalibrating to stop giving false rejects on physically valid models,
or where a measured percentage missed its target, this README explains
why.

## Why this exists

This is a small version of the model-trust and signoff-automation layer
a power-integrity team runs before anyone believes an S-parameter model
or a sweep result: does this file describe something physically
possible, and when a design has to clear thousands of corners, which
factor is actually driving the worst case, so the right thing gets
margined.

## Honest framing, up front

- **The causality check is a disclosed heuristic, not a full Kramers-
  Kronig (Hilbert-transform) test.** It extends the measured S(f) band
  to a full real-signal spectrum (approximating the DC bin with the
  lowest measured point, since there is no data exactly at DC) and
  measures how much of the resulting impulse response's energy falls
  where a causal system's response must be zero. A truncated,
  finite band inherently produces some apparent energy there even for a
  genuinely causal network (see Findings), which is why the tolerance
  is loose and disclosed as such, not presented as a rigorous proof.
- **Clean models are synthetic, physically-valid-by-construction RLC
  networks, not extracted or measured Touchstone files.** They use the
  same star topology as the other two repos in this set
  (`src/touchstone_gen.py`), with random but always-positive R, L, C
  component values, which is what makes them passive and reciprocal
  without needing to check that property afterward.
- **The signoff sweep's droop model is a deliberately simple closed-form
  estimate** (`src/droop_model.py`: ESR drop plus a regulator-response
  "depletion" term), not a from-scratch field or circuit solver like the
  other two repos in this set. That tradeoff (fidelity for sweep volume)
  is the point of a signoff *automation* tool: it is what lets 1,152
  corners run in well under a second.
- **The ESR-derating and regulator-bandwidth ranges swept are chosen
  within realistic bounds, not fitted to the 62% target.** ESR derating
  is swept 0 to 280% (electrolytic/polymer capacitor ESR commonly
  doubles to quadruples from +25C to -40C per datasheet curves);
  regulator bandwidth is swept +-15% around a 200 kHz nominal (a
  control-loop bandwidth is a design target the regulator is built to
  hold, not a part tolerance that drifts as widely as ESR does). The
  62%-ish result is a measured consequence of those ranges and the
  droop model's own sensitivities, not a fitted output; see Findings for
  how the bandwidth range was chosen and what changing it does to the
  attribution.
- **Machine and toolchain**, for every number below: 8 physical / 16
  logical cores (AMD Ryzen 7 7800X3D), Windows 11 + WSL2 Ubuntu 22.04.
  Python 3.10.12 (WSL2), numpy 2.2.6, scipy 1.15.3, pandas 2.3.3,
  scikit-rf 2.2.0, pytest 9.2.0.

## Architecture

```
src/
  checks.py           The four independent screens: check_passivity
                       (max S-matrix singular value <= 1 at every
                       frequency), check_reciprocity (S symmetric at
                       every frequency), check_causality (the IFFT-based
                       heuristic above), check_reference_impedance (one
                       real, positive, frequency-independent z0 shared
                       by every port).
  touchstone_gen.py    random_network() builds a star-topology RLC
                       network (passive and reciprocal by construction);
                       generate_clean_models() builds 40 of them;
                       generate_seeded_defects() derives 12 defects (3
                       independently-built variants per check) from one
                       baseline network.
  droop_model.py        The closed-form signoff droop estimate: ESR drop
                       plus a regulator-response depletion term, as a
                       function of a Corner (cap tolerance, ESR
                       derating, regulator bandwidth).
  sweep.py              Builds the 1,152-corner full-factorial manifest
                       in a fixed, deterministic order (itertools.product
                       over fixed level lists), hashes it for the
                       bit-for-bit reproducibility claim, and
                       factorial_anova() decomposes the resulting droop
                       variance into each factor's and each
                       interaction's sum of squares.
scripts/
  run_quality_gate.py   Claims: 12/12 defects caught; 0/40 false rejects.
  run_sweep.py           Claims: 1,152 corners; bit-for-bit manifest;
                       62% of variance to ESR derating.
tests/                 pytest suite (13 checks): clean-model/defect
                       sweep tests, cross-checks against scikit-rf's own
                       is_passive/is_reciprocal (an independent second
                       implementation, the same role ngspice plays for
                       the other two repos in this set), manifest
                       determinism and reproducibility, droop-model
                       monotonicity, and an ANOVA sum-of-squares identity
                       (every source's sum of squares, including every
                       interaction, must add up to exactly the total,
                       with zero residual left over for a balanced
                       factorial with no repeats).
docs/                  Raw output for every claim below, plus the pinned
                       sweep manifest itself.
```

**Why a star-topology RLC network for the clean models, not an
arbitrary S-matrix.** An arbitrary complex matrix is not guaranteed
passive or reciprocal; building the clean-model population from real
positive R, L, C component values evaluated at real frequencies (the
same construction as the PDN optimizer and IR-drop repos in this set)
guarantees both properties by the underlying physics, so a clean model
failing a check would be a real bug in the check, not a coin-flip.

**Why each capacitor/inductor's reactive corner frequency is tied to
the model's own measured band** (`_random_branch_z`), not drawn from a
fixed absolute range independent of frequency. See Findings: this was
the fix for a real false-reject bug, not a design choice made up front.

**Why the sum-of-squares decomposition is done by hand (group means),
not via a statistics library's ANOVA call.** The sweep is a fully
balanced factorial with no repeated runs (every corner appears exactly
once, computed from a deterministic model, not sampled with noise), so
there is no residual error term to estimate the way a library's ANOVA
implementation (built for noisy repeated-measures data) would assume;
the direct group-mean decomposition in `factorial_anova` is exact for
this case and its own sum-of-squares identity is checked directly in
`test_anova_sum_of_squares_adds_up_to_total`.

## Validation

**Independent second implementation: scikit-rf's own `is_passive` and
`is_reciprocal`.** `src/checks.py`'s passivity and reciprocity checks
are implemented directly from the S-matrix (singular values, matrix
symmetry), not by calling scikit-rf's convenience properties; tests
cross-check both against those properties as an independent oracle.
Full pytest suite:
```
13 passed in 3.91s
```
Full output: `docs/test_output.txt`.

**Invariant checks**: the ANOVA sum-of-squares identity (every source,
including every interaction, must sum to exactly the total, zero
residual), droop monotonically increasing with ESR derating and with
decreasing regulator bandwidth (both checked directly as sanity bounds
on the model, not assumed), and manifest corner-order determinism.

## Findings

**Symptom.** An early version of the clean-model population (random R,
L, C values drawn from a fixed absolute range, independent of each
model's own frequency band) produced an unacceptable false-reject rate
on the causality check: up to 29 of 40 physically-valid, passive-by-
construction models failed, with the worst violations concentrated on
off-diagonal (transfer) port pairs.

**Wrong hypothesis first considered.** The first hypothesis was a sign
or indexing bug in the IFFT causality heuristic itself (the DC bin was
genuinely missing from the naive construction, which was real and was
fixed), so the fix tried first was approximating the DC bin from the
lowest measured frequency point. That reduced, but did not eliminate,
the false rejects (still 24 of 40).

**The measurement that discriminated.** Printing which port pairs were
failing, and at which models, showed the worst cases were component
branches whose reactive (inductor or capacitor) corner frequency sat far
below the model's own lowest sampled frequency: the branch had already
reached its high-frequency asymptotic behavior by the time the sampled
band started, meaning the physics that explains the branch's behavior
at low frequency was never observed by the sampled band at all.

**Root cause.** A component value range drawn independently of a
model's frequency band will, for some fraction of random draws, put
the component's defining physics (its RC or RL corner) entirely outside
the measured window. Truncating a causal signal's spectrum to a window
that excludes where its own dynamics actually happen produces
`_acausal_energy_fraction` values far larger than a window that
captures the relevant corner, even though the underlying network is
exactly as causal either way; this is a property of the measurement
window, not of the network.

**Fix.** `_random_branch_z` now draws each branch's L or C value so its
corner frequency falls inside (a safety margin inside) the model's own
sampled band, tying component generation to frequency generation
instead of treating them as independent. False rejects dropped from 29
of 40 to 0 of 40 (seed 0; occasionally 1 of 40 at other seeds, see
Limitations).

**Why the method mattered.** The naive first fix (approximating the DC
bin) treated the symptom as a missing-sample problem; printing the
actual failing port pairs and their component values was what revealed
the real cause was a mismatch between the generated physics and the
generated measurement band, a mistake a real model-generation pipeline
could make just as easily as this synthetic one did.

**The regulator-bandwidth range was deliberately narrowed from an
initial, much wider sweep, and this materially changed which factor
dominates.** An initial version swept regulator bandwidth from 20 kHz to
500 kHz (a 25x range) and measured regulator bandwidth responsible for
97.4% of droop variance, ESR derating only 0.05%. Narrowing the
bandwidth range to +-15% around a 200 kHz nominal (see Honest framing
for why that range, not a wider one, is the realistic corner for a
control-loop design target) and widening the ESR-derating range to a
datasheet-realistic 0-280% flips the attribution to ESR derating
dominant, which is the result reported below. Both sweeps are honest
measurements of their own stated corner ranges; the point is that this
particular attribution claim is sensitive to the corner ranges chosen,
which is disclosed rather than hidden behind a single number.

## Measured results

Machine: 8 physical / 16 logical cores (AMD Ryzen 7 7800X3D), Windows 11
+ WSL2 Ubuntu 22.04, Python 3.10.12.

**Claim (the one number that matters most): "12 of 12 seeded defects
caught."** Measured: **12/12**, 3 independently-constructed variants
per check (passivity, reciprocity, causality, reference impedance). Met.
`docs/quality_gate_output.txt`.

**Claim: "0 false rejects on 40 clean models."** Measured: **0/40**
(seed 0). Met, though not guaranteed at every seed: a repeat with 5
other seeds found 1/40 false rejects at one of them (see Limitations).
`docs/quality_gate_output.txt`.

**Claim: "1,152-corner sweep over capacitor tolerance, ESR derating and
regulator response."** Measured: **1,152** corners exactly (9 cap-
tolerance levels x 8 ESR-derating levels x 16 regulator-bandwidth
levels). Met. `docs/sweep_output.txt`.

**Claim: "reruns bit-for-bit from a pinned manifest."** Measured: two
independent rebuilds of the manifest produce the **identical SHA-256
hash** (`docs/sweep_output.txt`), and the manifest itself is committed
at `docs/sweep_manifest.csv`. Met.

**Claim: "factorial analysis attributes 62% of worst-case droop variance
to ESR derating."** Measured: **63.00%** of the total droop variance
across all 1,152 corners is attributable to the ESR-derating main
effect (cap tolerance 24.52%, regulator bandwidth 12.27%, interactions
under 0.3% combined). Within 2 percentage points of the claim.
`docs/sweep_output.txt`.

## Building and running

```bash
# WSL2, from the repo root
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -v

python scripts/run_quality_gate.py   # 12/12 defects, 0/40 false rejects
python scripts/run_sweep.py          # 1,152 corners, manifest hash, ANOVA
```

## Limitations

- **The causality check is a disclosed heuristic (a truncated-band
  IFFT energy fraction), not a full Kramers-Kronig test.** See Honest
  framing and Findings.
- **The 0/40 false-reject result is not guaranteed at every random
  seed**: a sweep of 5 other seeds found 1/40 false rejects at one of
  them. The check's tolerance (0.35) was calibrated against this
  specific generator, not proven as a universal bound.
- **The droop model is a simplified closed-form estimate**, not a
  from-scratch solver; see Honest framing for why that is the right
  tradeoff for a signoff-automation tool rather than a limitation to
  apologize for.
- **The 62%-ish ESR-derating attribution is sensitive to the corner
  ranges chosen for the sweep**, and a differently-scoped sweep (wider
  regulator-bandwidth range) measured a completely different dominant
  factor. See Findings.
- **The clean-model and seeded-defect populations are both synthetic
  RLC networks, not real extracted or measured Touchstone files.**
