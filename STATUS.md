# nD-RoPE reproduction status

Status is authoritative for the checked-in evidence on `main`.

- Repository: `MachineLearning-Nerd/icml26-nd-rope-translation-invariance`
- Former repository: `MachineLearning-Nerd/icml26-repro-YPXDQkU7XW-nd-rope-translation-invariance`
- Paper: *nD-RoPE: A Generalized RoPE for n-Dimensional Position Embedding*
- Paper identifier: arXiv `2606.12146`; OpenReview `YPXDQkU7XW`
- Official implementation: `BoyangL1/nD-RoPE@f2cae70760806451f5e58be4b7e3dc4d0d856a1e`
- Overall gate: `SCOPED_PASS` — six claim contracts, raw evidence, 52-check verifier, six mutation controls, and release gate pass.
- Claim status: C1 `VERIFIED`; C2 `VERIFIED`; C3 `BLOCKED`; C4 `BLOCKED`; C5 `FALSIFIED_AS_WRITTEN`; C6 `FALSIFIED_AS_WRITTEN`.
- External evaluator snapshot: `6/12`, retrieved 2026-07-23 and preserved in `.openresearch/artifacts/source/live_verdict.json`; no new score is claimed.
- Compute boundary: local Apple M2 CPU; no GPU, paid remote compute, private checkpoint, or private token used.
- Canonical command: `uv run --frozen python repro/src/run_campaign.py`
- Canonical checker: `python repro/src/publication_gate.py --skip-producers`

The exact claim boundaries and limitations are in
[`docs/CLAIM_EVIDENCE.md`](docs/CLAIM_EVIDENCE.md) and
[`docs/SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md). The evaluator bundle under
`release/hf-space-candidate/` is archival and may contain historical Trackio
markup; it is not the source of the local verdicts.
