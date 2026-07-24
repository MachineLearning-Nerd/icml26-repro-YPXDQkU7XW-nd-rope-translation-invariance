# Claim 3 — current verification

**Verdict: BLOCKED (LOW confidence).** Table 1 claims that on all 50,000
ImageNet-1K validation images, under a 400-epoch matched ViT-S/DeiT-S
patch-16, 224×224 protocol, nD-RoPE scores `81.07%` top-1 versus
RoPE-Axial `80.89%` and RoPE-Mixed `80.90%`.

No proxy is treated as evidence for those numbers.

Paper source: [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2606.12146),
retrieved 2026-07-23, SHA-256
`aa7acd683cbb804762be15b607c1862b7be3fd0acd44424c939e2da9f0f6756f`;
anchors Table 1, Section 5, and Appendix C.

## Assumption audit and four routes

| Route | Different route | Observed evidence | Resolution |
| --- | --- | --- | --- |
| 1 | Artifact/data provenance | Pinned official release has 154 files, 0 checkpoints, 0 ImageNet data/prediction files | Unresolved |
| 2 | Architecture/training contract | nD width/head dimension `396/66`; axial and mixed `384/64`; no matched width | Confound, not falsification |
| 3 | Independent cross-table check | Table 1 nD-minus-mixed `+0.17`; Table 5 zero-degree `-0.18`; identical checkpoints/preprocessing not established | Diagnostic only |
| 4 | Exact-assumption falsification search | Rank reversal, random-model CPU test, and width confound each violate an assumption or lack full predictions | No valid counterexample |

The verification gate requires checkpoint digests, 50,000 labeled per-example
predictions, protocol metadata, and independently recomputed top-1 values.
Controls confirm that table transcription, partial predictions, and a missing
checkpoint digest are rejected; a complete synthetic record is accepted only
as a gate test, never as ImageNet evidence.

## Inspect the evidence

- [Exact claim contract](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim3/claim_contract.json)
- [Executable four-route source](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/repro/src/run_claim34.py)
- [Raw route report](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim34/claim34_report.json)
  and [raw controls](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim34/negative_controls.json)
- [Attempt-by-attempt record](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim3/attempts.md)
- [Independent checker output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json)
  and [per-claim nonzero mutation exit](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verifier_failure_controls.json)
- [Source audit](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim3/source_audit.md)
  and [method](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim3/method.md)

Fixed command: `uv run --frozen python repro/src/run_campaign.py` from
`current/`. Run SHA `{{GIT_SHA}}`; CPU/runtime are on the
[current verification page](#/index).

**Limitations, unblockers, and deviation.** Exact trained checkpoints, full ImageNet labels
and predictions, and a matched architecture or documented width-396
justification are required. The experiment deliberately does not substitute
untrained models, subsets, or another dataset.
