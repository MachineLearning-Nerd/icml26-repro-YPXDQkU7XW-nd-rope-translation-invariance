# Claim 4 — current verification

**Verdict: BLOCKED (LOW confidence).** Table 5 and Section 5.2 claim that,
with the exact already-trained fixed ImageNet models and no fine-tuning, all
50,000 validation images resized to 256, rotated 30 degrees, and center-cropped
to 224 give nD-RoPE `78.51%` top-1 versus RoPE-Mixed `71.34%`.

Paper source: [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2606.12146),
retrieved 2026-07-23, SHA-256
`aa7acd683cbb804762be15b607c1862b7be3fd0acd44424c939e2da9f0f6756f`;
anchors Table 5 and Section 5.2.

## Assumption audit and four routes

| Route | Different route | Observed evidence | Resolution |
| --- | --- | --- | --- |
| 1 | Artifact/data provenance | 0 fixed checkpoints and 0 full prediction vectors in the 154-file release | Unresolved |
| 2 | Source/protocol audit | No rotation evaluator; model widths differ `396` vs `384` | Confound only |
| 3 | Transform/cross-table audit | Resize interpolation, rotation interpolation, fill, and antialias are unspecified; zero-degree ranking reverses across tables | Diagnostic only |
| 4 | Exact falsification search | Random models, cross-table comparison, and width mismatch do not preserve the exact fixed-model protocol | No valid counterexample |

The fail-closed gate requires both exact checkpoint digests, all 50,000 labels
and predictions, and a complete deterministic transform record. Table values
alone, partial predictions, and missing checkpoint hashes are negative controls
that must fail.

## Inspect the evidence

- [Exact claim contract](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim4/claim_contract.json)
- [Executable four-route source](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/repro/src/run_claim34.py)
- [Raw route report](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim34/claim34_report.json)
  and [raw negative controls](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim34/negative_controls.json)
- [Attempt-by-attempt record](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim4/attempts.md)
- [Independent checker output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json)
  and [per-claim nonzero mutation exit](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verifier_failure_controls.json)
- [Source audit](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim4/source_audit.md)
  and [method](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim4/method.md)

Fixed command: `uv run --frozen python repro/src/run_campaign.py` from
`current/`. Run SHA `c3283858f04836356ce806a7b884a228c5cfb954`; CPU/runtime are on the
[current verification page](#/index).

**Limitations, unblockers, and deviation.** The exact Table 5 checkpoints, ImageNet-1K
validation access, and complete interpolation/fill/antialias definition are
required. No untrained or downscaled rotation test is promoted.
