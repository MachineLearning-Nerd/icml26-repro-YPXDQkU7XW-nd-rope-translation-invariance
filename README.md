# nD-RoPE reproduction: claim-by-claim evidence

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-YPXDQkU7XW-nd-rope-translation-invariance/blob/master/notebooks/ndrope_reproduction.py)

This project reproduces and audits
**[nD-RoPE: A Generalized RoPE for n-Dimensional Position
Embedding](https://arxiv.org/abs/2606.12146)**. The strongest new result is a
direct counterexample to Claim 6: at the paper’s stated 2,048-point training
grid, Table 7 reports **85.80 mIoU for θ=2** and **85.58 for θ=100**, contrary
to the text’s claim that θ=100 is best “across all settings.”

The cumulative result is:

- Claims 1 and 2: **VERIFIED** by numerical certificates and pinned-code parity.
- Claims 3 and 4: **BLOCKED** after four distinct routes; the exact trained
  ImageNet checkpoints, predictions, and complete rotation protocol are absent.
- Claim 5: **FALSIFIED** as written; the released 85.97-mIoU path is
  ShapeNetPart, not ModelNet40, and SemanticKITTI code is absent.
- Claim 6: **FALSIFIED** by the exact paper-table counterexample above.

The live judge score remains **6/12**. A conservative post-publication forecast
is **6–8/12**; **8/12** is the best-supported possible score if the live judge
accepts Claim 6. These are forecasts, not awarded points.

The approved 12-file text release is published to the existing Hugging Face
Space at revision
[`f457f54c89151cc850279e904d28956e4c23508b`](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/commit/f457f54c89151cc850279e904d28956e4c23508b).
An exact-revision download confirmed all approved hashes, all 20 protected
paths, and byte identity for every protected page. The live judge has queued
this revision for re-evaluation; the score remains 6/12 until that evaluation
finishes.

Formal runs used local Apple M2 CPU compute and the locked `uv` environment.
No GPU or paid Hugging Face cpu-upgrade was used. The ImageNet claims were not
downscaled or replaced with proxies: they remain blocked because a smaller or
random-model test would not satisfy their assumptions.

Read the [illustrated technical report](reports/ndrope-reproduction/report.md)
and [publication release record](reports/ndrope-reproduction/release-record.md),
or explore the
[self-contained marimo tutorial](notebooks/ndrope_reproduction.py). The
notebook can be opened with the Molab badge above and embeds the headline
values so readers need not rerun expensive work.

## Experiment log

Every formal node uses the exact inherited command
`uv run --frozen python repro/src/run_campaign.py`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
| --- | --- | --- | --- | --- |
| [`orx/baseline-judged-6-12-evidence`](https://github.com/MachineLearning-Nerd/icml26-repro-YPXDQkU7XW-nd-rope-translation-invariance/tree/orx/baseline-judged-6-12-evidence) | Frozen judged evidence baseline | `uv run --frozen python repro/src/run_campaign.py` | C1/C2 verified; C5 falsified; regression passed | local Apple M2 CPU |
| [`orx/claim-6-exact-table-contradiction`](https://github.com/MachineLearning-Nerd/icml26-repro-YPXDQkU7XW-nd-rope-translation-invariance/tree/orx/claim-6-exact-table-contradiction) | Test exact Table 6–8 contracts | `uv run --frozen python repro/src/run_campaign.py` | C6 falsified by Table 7 at 2,048 points | local Apple M2 CPU |
| [`orx/claims-3-4-route-1-artifact-provenance`](https://github.com/MachineLearning-Nerd/icml26-repro-YPXDQkU7XW-nd-rope-translation-invariance/tree/orx/claims-3-4-route-1-artifact-provenance) | Complete release/provenance inventory | `uv run --frozen python repro/src/run_campaign.py` | No trained checkpoint or prediction evidence | local Apple M2 CPU |
| [`orx/claims-3-4-route-2-architecture-contract`](https://github.com/MachineLearning-Nerd/icml26-repro-YPXDQkU7XW-nd-rope-translation-invariance/tree/orx/claims-3-4-route-2-architecture-contract) | Recover comparison architecture | `uv run --frozen python repro/src/run_campaign.py` | Width 396/66 vs 384/64 confound | local Apple M2 CPU |
| [`orx/claims-3-4-route-3-cross-table-protocol`](https://github.com/MachineLearning-Nerd/icml26-repro-YPXDQkU7XW-nd-rope-translation-invariance/tree/orx/claims-3-4-route-3-cross-table-protocol) | Cross-table/protocol diagnostic | `uv run --frozen python repro/src/run_campaign.py` | Rank reversal found; not a valid falsification | local Apple M2 CPU |
| [`orx/claims-3-4-route-4-falsification-search`](https://github.com/MachineLearning-Nerd/icml26-repro-YPXDQkU7XW-nd-rope-translation-invariance/tree/orx/claims-3-4-route-4-falsification-search) | Mandatory assumption-preserving falsification route | `uv run --frozen python repro/src/run_campaign.py` | C3/C4 BLOCKED after four routes | local Apple M2 CPU |
| [`orx/release-candidate-evidence-and-report`](https://github.com/MachineLearning-Nerd/icml26-repro-YPXDQkU7XW-nd-rope-translation-invariance/tree/orx/release-candidate-evidence-and-report) | Cumulative evidence, report, notebook, release gate | `uv run --frozen python repro/src/run_campaign.py` | Passed 22 tests, 35 checks, and the additive release gate | local Apple M2 CPU |
| [`orx/final-approval-candidate`](https://github.com/MachineLearning-Nerd/icml26-repro-YPXDQkU7XW-nd-rope-translation-invariance/tree/orx/final-approval-candidate) | Package the exact successful evidence and rerun before approval | `uv run --frozen python repro/src/run_campaign.py` | Passed 22 tests, 35 checks, and the release gate | local Apple M2 CPU |
| `master` | Publication surface | Not run as an experiment (publication surface) | Published reproduction surface; not a formal experiment | N/A |

## Reproduce

```bash
uv sync --frozen
uv run --frozen python repro/src/run_campaign.py
```

The fixed runner regenerates raw CSV/JSON evidence, negative controls,
independent checks, the five report figures, notebook validation, and the
protected Hugging Face Space release gate. Claim contracts, methods, source
audits, `EVAL.md` files, and raw outputs are under
`.openresearch/artifacts/`.

The paper source is pinned by URL, retrieval date, anchors, and SHA-256. The
author implementation is pinned to
[`BoyangL1/nD-RoPE@f2cae707`](https://github.com/BoyangL1/nD-RoPE/tree/f2cae70760806451f5e58be4b7e3dc4d0d856a1e).
No paper-table number is presented as an independently trained reproduction
unless its underlying experiment was actually run.
