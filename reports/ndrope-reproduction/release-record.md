Previous live judged score: `6/12`

Conservative projected score range after the published change: **6–8/12**

Best-supported possible new score: **8/12 (forecast, not a judge result)**

# Publication and post-publication verification record

The approved evaluator-visible reproduction was published to the existing
Space `DineshAI/YPXDQkU7XW` at exact revision
[`458c9256e6b04cb2752c1fc20efe4094074a5283`](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/commit/458c9256e6b04cb2752c1fc20efe4094074a5283).
No second Space was created, no protected evidence was deleted, and no GPU or
paid compute was used.

The published verdict vector is `VERIFIED`, `VERIFIED`, `BLOCKED`, `BLOCKED`,
`FALSIFIED`, `FALSIFIED`. The live score remains **6/12** until the judge
evaluates the new revision. No score increase is claimed here.

## Claim forecast

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
| --- | --- | --- | --- | --- | --- |
| 1 | 2/2 | 2/2 | HIGH | VERIFIED | An independently reconstructed symbolic derivation covers the all-real-phase quantifier. The accepted 7,680 rotary and 192 Fourier trials, official-code parity, raw data, checker, and controls remain visible as corroboration. |
| 2 | 2/2 | 2/2 | HIGH | VERIFIED | The all-`n` simplex Gram/rank/tight-frame derivation and unique economy optimum at `rho=e` cover the universal statement; 7,936 symmetry trials and 64 independent optimizations corroborate it. |
| 3 | 0/2 | 0/2 | LOW | BLOCKED | Four materially different routes, including the mandatory exact falsification search, are complete. No trained ImageNet checkpoint or 50,000-image predictions exist; the width mismatch is a confound, not a metric counterexample. |
| 4 | 0/2 | 0/2 | LOW | BLOCKED | Four routes, including mandatory falsification, are complete. Exact fixed checkpoints and interpolation/fill/antialias details remain unavailable; no proxy is promoted. |
| 5 | 2/2 | 2/2 | HIGH | FALSIFIED | The pinned 154-file release shows the 85.97-mIoU entrypoints execute ShapeNetPart with 16 categories and 50 part labels, never the ModelNet loader; SemanticKITTI code is absent. |
| 6 | 0/2 | 2/2 | HIGH | FALSIFIED | Two independent dynamic CPU counters execute the exact released 224×224 models. Attention MACs rise 5.69%; a matched-width control explains 99.81%/99.97% of the measured delta; live enumeration finds zero trainable frequency parameters. Remaining risk is judge interpretation of Appendix D.4. |

Current total score: **6/12**. Conservative projected total: **6–8/12**.
Best-supported possible total: **8/12**. Only the live judge can change the
score.

## Direct response to judge feedback

The judge rejected the previous Claim 6 route because it recomputed
paper-table means without executing an ablation or profiling a model. The new
route executes the exact released image architectures:

| Architecture | `torch.profiler` GMAC | Independent dispatch GMAC |
| --- | ---: | ---: |
| baseline, width 384 | 4.600286208 | 4.598882304 |
| nD-RoPE, width 396 | 4.8793826215 | 4.877486784 |
| matched-width baseline, width 396 | 4.878849888 | 4.877402112 |
| monotonic control, width 408 | 5.165583552 | 5.164091904 |

The measured increases are 6.06694% and 6.05809%. A separate symbolic
attribution gives 5.68977% extra attention MACs but is not used circularly as
the measurement. The matched-width baseline explains 99.81% and 99.97% of the
official-model delta. This contradicts Appendix D.4's statement that image
nD-RoPE only changes frequency construction “without introducing additional
attention cost.” The live model has zero trainable frequency parameters.

Claims 1 and 2 were also upgraded to satisfy universal-theorem calibration:
their symbolic derivations cover the full quantifiers, while finite sweeps are
explicitly labeled corroboration.

## Experiment tree and immutable revisions

| Branch | Commit | Formal run | Outcome |
| --- | --- | --- | --- |
| `orx/evaluator-visible-current-verification` | `c3283858f04836356ce806a7b884a228c5cfb954` | `ea50d3f3-5c16-4bda-8915-188ff91cd847` | 24 tests, 52 checks, six mutation exits, and all release checks pass |
| `orx/blind-review-candidate-package` | `f38ab2213b1f5f27caed0117326c1f5f3362be70` | `ec7b9c8c-144d-488d-ace5-f150a5aab5db` | Exact frozen candidate independently reruns and passes |

Winning publication branch: `orx/blind-review-candidate-package`.

Scientific evidence source SHA:
`c3283858f04836356ce806a7b884a228c5cfb954`.

Frozen candidate SHA:
`f38ab2213b1f5f27caed0117326c1f5f3362be70`.

Published Hugging Face SHA:
`458c9256e6b04cb2752c1fc20efe4094074a5283`.

## Commands, environment, and compute

The fixed command inherited unchanged by every experiment node was:

```text
uv run --frozen python repro/src/run_campaign.py
```

The final formal launches were:

```text
orx exp run 7e5378c0-ae89-4987-bb53-ce2645e77762 --backend local
orx exp run 2f6bea26-61e7-44c1-95d7-c4b4efd4f803 --backend local
```

The fixed runner executes Claims 1 and 2, the source audit, Claim 6's exact
model profilers, Claims 3/4's four-route audit, symbolic proofs, 24 tests, six
verifier-mutation cases, the 52-check independent verifier, report generation,
strict marimo validation, evaluator-bundle generation, and the release gate.

Compute was an Apple M2 CPU with eight logical CPUs and one locked
repository-level `.venv` (Python 3.12.11 and `uv`). The two final formal runs
took 131.85 and 156.46 seconds. GPU use: none. Hugging Face cpu-upgrade: not
used. Local and Hugging Face compute cost: `$0`.

## Release and evaluator-visible gates

- Candidate files: 309.
- Protected Space files retained: 31/31.
- Historical pages byte-identical before upload: 14/14.
- Self-contained current bundle: 269 UTF-8 files.
- Exact changed/new text upload allowlist: 280 files.
- Upload allowlist SHA-256:
  `bdd4e7c53a5551c5164731dc01e4267edef32feaccced0185cd209428026a0a3`.
- Upload manifest SHA-256:
  `a133be3c544ba3a858a57591ebe602c3fd535b54ad02c8028134337c7c67fd17`.
- Current verification is the canonical default; the old rejected verifier is
  labeled “Historical rejected baseline.”
- Independent checks: 52/52 pass.
- Per-claim mutation cases: 6/6 exit nonzero.
- Secret-pattern scan: no hits.

After publication, the exact HF revision was downloaded into an empty
directory. Verification matched all 309 candidate paths and hashes, all 280
allowlisted upload hashes, and all 29 immutable protected hashes. `README.md`
and `logbook.json` were the only approved mutable protected paths and matched
their upload-manifest hashes.

Starting only at the canonical current verification page, the post-publication
blind traversal opened 62 files through 54 visible local links. It located the
exact claim and source quantifiers, assumptions, executable source, fixed
command, pinned environment, inline data, raw downloads, independent checker,
negative control, limitations, Git SHA, seeds, CPU/runtime metadata, and
nonzero mutation exit for every claim.

The exact published text tree is preserved under
`release/hf-space-candidate/`. The line-by-line publication boundary is
`release/hf-space-upload-allowlist.txt`, and the per-file hashes are in
`release/hf-space-upload-manifest.sha256`.

## Awaiting live judge

Revision `458c9256e6b04cb2752c1fc20efe4094074a5283` is the current Space head
and is ready for live evaluation. There is no separate local `orx` command for
changing judge state; publishing a new Space head is the queueing event. This
record therefore reports the immutable revision and waits for the evaluator
without claiming a score change.
