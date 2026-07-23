Previous live judged score: `6/12`

Conservative projected score range after the proposed change: **6–8/12**

Best-supported possible new score: **8/12 (forecast, not a judge result)**

# Publication release record

The reproduction campaign reached the honest release gate, and the user
approved the exact 12-file text payload. It was published to the existing
Space `DineshAI/YPXDQkU7XW` at revision
`f457f54c89151cc850279e904d28956e4c23508b`. An exact-revision download
verified the approved hashes, the complete 31-file tree, preservation of all
20 protected paths, and byte identity for every protected page. The live
judge queued that revision for re-evaluation. The reader-facing reproduction
surface is mirrored to GitHub `master`; the live score remains 6/12 until the
judge evaluates the new Space revision.

## Claim forecast

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
| --- | --- | --- | --- | --- | --- |
| 1 | 2/2 | 2/2 | HIGH | VERIFIED | 7,680 rotary trials and 192 Fourier trials pass; maximum translation error 1.42e-14; pinned-author rotation parity 1.29e-7. Risk: numerical certificates complement rather than replace the proof. |
| 2 | 2/2 | 2/2 | HIGH | VERIFIED | Dimensions 2–32, 7,936 symmetry trials, 64 independent optimizations, and malformed-geometry controls pass. Maximum structural error 1.99e-15; optimum error 4.44e-8. |
| 3 | 0/2 | 0/2 | LOW | BLOCKED | Four routes completed: artifact provenance, architecture contract, cross-table protocol, and mandatory exact falsification search. No trained checkpoint or full ImageNet predictions exist; width 396/66 versus 384/64 is a confound, not a metric counterexample. |
| 4 | 0/2 | 0/2 | LOW | BLOCKED | Four routes completed, including mandatory falsification. Exact Table 5 checkpoints and resize/rotation interpolation, fill, and antialias settings remain unavailable. |
| 5 | 2/2 | 2/2 | HIGH | FALSIFIED | Pinned release rerun confirms the 85.97-mIoU entrypoint is ShapeNetPart with 16 categories and 50 part labels, not ModelNet40; SemanticKITTI code is absent. |
| 6 | 0/2 | 2/2 | HIGH | FALSIFIED | Appendix D.3’s “best across all settings” statement has a strict in-domain counterexample in Table 7 at the stated 2,048-point training grid: θ=2 is 85.80 versus θ=100 at 85.58 (+0.22 points). Remaining risk: the judge may interpret the prose non-universally; no ablation checkpoint exists to retrain. |

Current total score: **6/12**. Conservative projected total: **6–8/12**.
Best-supported possible total: **8/12**. Only the live judge can change the
score.

Since the previous verdict, Claim 6 changed from inconclusive to a
high-confidence falsification. Claims 3 and 4 changed from undifferentiated
inconclusive states to rigorously documented blockers after the required four
routes, but no points are forecast for them. Claims 1, 2, and 5 are unchanged
and pass the cumulative regression suite.

## Final claim results

- **Claim 1 — VERIFIED.**
- **Claim 2 — VERIFIED.**
- **Claim 3 — BLOCKED.** Unblocked by exact nD/Axial/Mixed checkpoints,
  ImageNet-1K validation labels and per-example predictions, and a matched
  architecture or documented width-396 justification.
- **Claim 4 — BLOCKED.** Unblocked by the exact Table 5 checkpoints,
  ImageNet-1K validation access, and complete resize/rotation settings.
- **Claim 5 — FALSIFIED.**
- **Claim 6 — FALSIFIED.**

No toy, proxy, skipped, missing-data, or random-model result is labeled
full-scale evidence.

## Source and environment

- Paper: `2606.12146`, retrieved from
  `https://ar5iv.labs.arxiv.org/html/2606.12146` with an explicit browser
  User-Agent on `2026-07-23T16:21:35Z`.
- Paper SHA-256:
  `aa7acd683cbb804762be15b607c1862b7be3fd0acd44424c939e2da9f0f6756f`.
- Anchors: Section 4.1/Equation 6; Sections 4.2–4.3 and Appendix E/Equations
  18–19; Tables 1 and 5; Tables 2–4; Appendix D.3/Tables 6–8.
- Author source: `BoyangL1/nD-RoPE` commit
  `f2cae70760806451f5e58be4b7e3dc4d0d856a1e`, complete 154-file manifest.
- Baseline repository SHA:
  `9cd7aa764e62364b9220830b5b7ea7523f3c3a9d`.
- One repository `.venv`, Python 3.12.11, uv 0.11.29, locked by
  `pyproject.toml`, `.python-version`, and `uv.lock`.
- Final lock SHA-256:
  `9a62d75a6576c753a2c059c627f1d3bfd8973f7caaf08bf695a40bc0bffbcbca`.
- Live verdict dataset head at the approval gate:
  `4feef54d43b176732d46fb04a6367506c2fe7b42`. Filtering the object by the
  exact key/`space_id` `DineshAI/YPXDQkU7XW` returns one record. Its judged
  timestamp, Space SHA, six verdicts, and resulting 6/12 score remain
  unchanged from the supplied live baseline.

## Experiment tree

| Branch | Commit | Role | Terminal duration | Result |
| --- | --- | --- | ---: | --- |
| `orx/baseline-judged-6-12-evidence` | `7eb190b` | frozen baseline | 50s | accepted Claims 1/2/5 pass |
| `orx/claim-6-exact-table-contradiction` | `a337736` | exact Claim 6 contract | 1m00s | Claim 6 FALSIFIED |
| `orx/claims-3-4-route-1-artifact-provenance` | `3551401` | artifact route | 55s | unresolved, gate fail-closed |
| `orx/claims-3-4-route-2-architecture-contract` | `8ae5f80` | source/architecture route | 1m10s | unresolved confound |
| `orx/claims-3-4-route-3-cross-table-protocol` | `c2f58c8` | independent table/protocol route | 1m35s | diagnostic only |
| `orx/claims-3-4-route-4-falsification-search` | `3f9f12c` | mandatory falsification route | 3m29s | Claims 3/4 BLOCKED |
| `orx/release-candidate-evidence-and-report` | `e9ee75c` | cumulative release preflight | 4m37s | all gates pass |
| `orx/final-approval-candidate` | `c1d4e3c` | final winning branch | 3m17s | all gates pass |

Winning branch: `orx/final-approval-candidate`.

Winning Git SHA:
`c1d4e3c0c062d4ff3d142a444dade0f91e12a1b4`.

The stacked tree descends from the frozen baseline through the Claim 6 branch,
the four ImageNet routes, the release candidate, and the final approval
candidate. No completed experiment branch was rebased or merged.

## Commands

The fixed command inherited unchanged by every experiment node was:

```text
uv run --frozen python repro/src/run_campaign.py
```

The final runner expanded it into:

```text
python repro/src/run_claim1.py --official-root vendor/nD-RoPE --output-dir outputs/claim1 --seeds 32 --trials-per-seed 240 --parseval-seeds 64 --scales 3 --theta 100.0
python repro/src/run_claim2.py --output-dir outputs/claim2 --max-dimension 32 --permutations-per-dimension 256
python repro/src/run_source_audit.py --official-root vendor/nD-RoPE --output-dir outputs/source_audit
python repro/src/run_claim6.py --data repro/data/paper_claim6.json --official-source vendor/nD-RoPE/rope-vit-ndrope/deit/models_v2_ndRope.py --output-dir outputs/claim6
python repro/src/run_claim34.py --data repro/data/paper_claim34.json --official-root vendor/nD-RoPE --output-dir outputs/claim34
python -m pytest -q repro/tests
python repro/src/verify_results.py --root .
python repro/src/generate_report_assets.py
python -m marimo check --strict notebooks/ndrope_reproduction.py
python repro/src/verify_release.py --root .
```

The orchestration commands were:

```text
orx exp run f1f18cba-a9c2-42eb-8087-469b00d78fbe --backend local
orx exp run 541f5438-2bda-40ee-b461-b84594e7513e --backend local
orx exp run 4bd7a547-adc0-4385-b528-b7d2274be441 --backend local
orx exp run 714ed8ef-f43c-4d97-be31-7ba0e0564798 --backend local
orx exp run 432bbc20-b019-4923-aa29-16ead7fa3126 --backend local
orx exp run 6989c775-0099-4ab4-bdf8-4a416b86a0e0 --backend local
orx exp run 181e4abb-164c-4872-8269-16f292ff0132 --backend local
orx exp run d436ce63-522c-45ad-9913-6f31a5ef152c --backend local
```

Each launch was followed by `orx exp wait`, terminal `orx runs`, and
`orx logs` inspection. Startup also ran `orx skill`, the
`orx-experiment-tree`, `orx-evidence`, `orx-git`, and `orx-compute` skill
guides, `orx projects --json`, project/run inspection, Git SHA/status checks,
disk inspection, and an environment-name-only inventory. Paper and verdict
inputs were downloaded at exact revisions and hashed before use.

## Reproducibility and release gate

Final run `6347793c-ade6-4325-a3c5-1181564a9232` completed on commit
`c1d4e3c` in 3m17s:

- 22/22 tests passed.
- All 35 independent scientific checks passed.
- Claims 1, 2, 5, and 6 passed cumulative regression.
- Claims 3 and 4 terminated as BLOCKED only after all four routes.
- All negative controls behaved as specified.
- Five report figures regenerated and passed PNG validation.
- `marimo check --strict` passed.
- Candidate logbook JSON and every tree path validated.
- No secret-pattern hits were found.

The baseline and final formal runs consumed 16m53s of aggregate local
experiment wall time across eight nodes. Hardware was an Apple M2 CPU with
eight logical CPUs and 16 GB RAM. Hugging Face cpu-upgrade was not needed.
GPU use: none. Local cost: `$0`. Hugging Face cost: `$0`.

## Protected Space subset proof

- Existing Space: `DineshAI/YPXDQkU7XW`.
- Protected HF Head:
  `dcbbd49ee4487f62443820c7dfe2be11ae10af51`.
- Protected Judge Head:
  `dcbbd49ee4487f62443820c7dfe2be11ae10af51`.
- Old manifest: 20 files.
- Candidate: 31 files.
- Old path set is a subset of the candidate path set: **PASS**.
- All 10 pre-existing Markdown pages are byte-identical: **PASS**.
- Unexpected changes to protected files: none.
- The sole intentionally changed old path is routing metadata
  `logbook.json`.
- New files: four pages plus seven evidence files.
- Upload is restricted to the changed routing JSON and 11 new text files.

## Exact Hugging Face upload allowlist

```text
logbook.json
pages/campaign-2026-07-23-summary/page.md
pages/claim-3-four-route-audit/page.md
pages/claim-4-four-route-audit/page.md
pages/claim-6-exact-counterexample/page.md
evidence/2026-07-23/EVAL.md
evidence/2026-07-23/claim34_negative_controls.json
evidence/2026-07-23/claim34_report.json
evidence/2026-07-23/claim6_negative_controls.json
evidence/2026-07-23/claim6_report.json
evidence/2026-07-23/run_metadata.json
evidence/2026-07-23/verification.json
```

The upload manifest SHA-256 is
`d27817adf75dfb96a206938dcf505e8c8a5f7590f6edbead11ffed6057c2f06d`.
The complete per-file SHA-256 manifest is saved at
`../../release/hf-space-upload-manifest.sha256`. The exact published text tree
is preserved under `../../release/hf-space-candidate/`.

## Evidence paths

- `report.md` — illustrated technical article.
- `images/` — five verified evidence-bearing figures.
- `../../EVAL.md` — final claim matrix.
- `../../outputs/verification.json` — 35 independent checks.
- `../../outputs/claim34/claim34_report.json` — all four Claims 3/4 routes.
- `../../outputs/claim6/claim6_report.json` — exact counterexample.
- `../../.openresearch/artifacts/run_metadata.json` — command, SHA, seeds,
  environment, and runtime.
- `../../.openresearch/artifacts/release/release_gate.json` — subset, logbook,
  image, allowlist, and secret
  checks.
- `../../release/hf-space-upload-allowlist.txt` and
  `../../release/hf-space-upload-manifest.sha256` — publication boundary.
- `../../release/hf-space-candidate/` — exact published text tree.

## Approved publication completed

Only the 12 allowlisted text files were committed to the existing Space
through the Hugging Face API, with the protected revision as the parent. The
resulting revision is
`f457f54c89151cc850279e904d28956e4c23508b`. No protected page changed, no
file was deleted, and no second Space was created. The exact revision was
downloaded and independently verified before the GitHub publication surface
was prepared. The judge state reports this revision as queued for re-judging.
No score increase is claimed.
