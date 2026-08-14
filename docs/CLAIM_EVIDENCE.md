# Claim-to-evidence map

Each status below is tied to a paper anchor, an executable producer, raw
outputs, an independent check, and a negative control. `VERIFIED` means the
scoped mathematical or source contract passes. `BLOCKED` means the exact paper
experiment cannot be tested from the public artifacts. `FALSIFIED_AS_WRITTEN`
means the checked contract has a reproducible contradiction while narrower
subclaims may remain untested.

| Claim | Paper target | Producer and evidence | Independent check / control | Status |
| --- | --- | --- | --- | --- |
| C1 | Section 4.1, including the translation-invariant Fourier factorization and Eq. 6 | `repro/src/run_symbolic_proofs.py` covers the universal real-phase identities; `repro/src/run_claim1.py` produces 7,680 rotary and 192 Parseval/Fourier cases under `outputs/claim1/`. | `repro/src/verify_results.py`; additive-position control; `outputs/symbolic_proof_certificates.json`; `outputs/verifier_failure_controls.json`. | `VERIFIED` |
| C2 | Section 4.2, Eqs. 18–19, and Appendix A.5 economy optimum | `repro/src/run_claim2.py` constructs dimensions 2–32, tests simplex geometry/permutations, and runs 64 independent economy optimizations under `outputs/claim2/`. | Rank, centroid, Gram, tight-frame, malformed-simplex, and off-optimum controls; independent checker in `outputs/verification.json`. | `VERIFIED` |
| C3 | Table 1 / Section 5.1 ImageNet-1K 224×224 top-1 values | `repro/src/run_claim34.py` executes four routes recorded in `outputs/claim34/` and `.openresearch/artifacts/claim3/`. The route requires the exact trained checkpoint, 50,000 validation predictions, and protocol metadata. | Provenance inventory, architecture-width audit, cross-table diagnostic, and exact assumption-preserving falsification search; no proxy model. | `BLOCKED` |
| C4 | Table 5 / Section 5.2 zero-shot 30° rotation values | The same four-route `run_claim34.py` campaign audits checkpoint/data availability, rotation protocol completeness, and valid counterexamples. | Evidence gate remains false; missing interpolation/fill/antialias and trained checkpoints are explicitly retained as limitations. | `BLOCKED` |
| C5 | Tables 2–4 / Section 5.3 cross-modal benchmarks | `repro/src/run_source_audit.py` inventories the pinned 154-file author release and inspects `train_partseg_ndrope.py`, `test_partseg.py`, and dataset loaders. | ShapeNetPart path, 16 categories, 50 part labels, absent SemanticKITTI code, and conjunctive-claim logic are checked in `outputs/source_audit/`. | `FALSIFIED_AS_WRITTEN` |
| C6 | Appendix A.4 ablations and Appendix D.4 computational-cost statement | `repro/src/run_claim6.py` recomputes table contracts and finds the exact 2,048-point `theta=2` versus `theta=100` contradiction. `repro/src/run_claim6_flops.py` executes the exact released 224×224 width-384/396 models and a width-396 control. | Two dynamic counters, symbolic attention-MAC calculation, width-408 monotonic control, frequency-buffer/parameter check, and mutation control. | `FALSIFIED_AS_WRITTEN` |

## Campaign and fail-closed verification

The campaign is dispatched by `repro/src/run_campaign.py`. Its final checked-in
run metadata records the fixed command, environment, seeds, CPU, runtime, and
source commit. `repro/src/verify_results.py` recomputes counts and assertions
from raw CSV/JSON rather than trusting displayed prose. The six cases in
`outputs/verifier_failure_controls.json` intentionally corrupt one decisive
claim value; every mutated run exits nonzero.

The evaluator-visible package under `release/hf-space-candidate/` is a frozen
publication snapshot. Its `current/` bundle mirrors the same claim contracts,
while the older pages are labeled historical. The local `.openresearch/` and
`outputs/` trees are canonical for this repository.
