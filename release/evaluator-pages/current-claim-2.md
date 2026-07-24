# Claim 2 — current verification

**Verdict: VERIFIED.** Sections 4.2–4.3, Appendix E, and Equations 18–19 claim
that for every `n≥2`, each scale uses `n+1` maximally symmetric regular-simplex
directions spanning rank `n`, and the economy objective uniquely gives
`r*=exp(1/n)`.

Paper source: [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2606.12146),
retrieved 2026-07-23, SHA-256
`aa7acd683cbb804762be15b607c1862b7be3fd0acd44424c939e2da9f0f6756f`;
anchors Sections 4.2–4.3, Equations 18–19, and Appendix E.

## Universal derivation

Assumptions: integer `n≥2`, `ρ>1`, and resolution `R>1`, under the paper's
geometric-ladder economy objective.

In the zero-sum `n`-dimensional subspace of `R^(n+1)`, define

```text
v_i = sqrt((n+1)/n) (e_i - 1/(n+1) 1).
```

Exact algebra gives, for every integer `n≥2`,

```text
sum_i v_i = 0
v_i·v_i = 1
v_i·v_j = -1/n  (i != j)
sum_i v_i v_i^T = ((n+1)/n) I  on the zero-sum subspace.
```

The Gram matrix therefore has one zero eigenvalue and `n` eigenvalues
`(n+1)/n`, proving rank coverage and tight-frame isotropy for the full
quantifier.

For `ρ=r^n>1` and resolution `R>1`, Appendix E's cost is proportional to
`E(ρ)=ρ log(R)/log(ρ)`. Its derivative is

```text
E'(ρ) = log(R) (log(ρ)-1) / log(ρ)^2.
```

It is negative on `(1,e)`, zero only at `e`, and positive on `(e,∞)`.
Thus `ρ*=e` is the unique global minimizer and `r*=exp(1/n)`. The
[exact symbolic source](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/repro/src/run_symbolic_proofs.py)
and [certificate](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/symbolic_proof_certificates.json)
make this proof, not finite examples, decisive.

## Observed corroboration

| Audit | Result |
| --- | ---: |
| Dimensions | 2–32 |
| Vertex-permutation trials | 7,936 |
| Independent economy optimizations | 64 |
| Maximum centroid error | `1.9877382144719217e-15` |
| Maximum Gram error | `9.749145934989656e-16` |
| Maximum tight-frame error | `1.5543122344752192e-15` |
| Maximum `r*` optimization error | `4.438399692219264e-8` |
| 3D base upper bound | `207.1272488898345` |

All 15 dropped, perturbed, and rank-deficient controls were detected.
For the displayed 3D bound, `θ=100` is inside while `10^4` and `10^6` are
outside.

## Inspect and rerun

- [Claim contract](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim2/claim_contract.json)
- [Executable Claim 2 source](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/repro/src/run_claim2.py)
- [Raw report](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim2/claim2_report.json),
  [geometry CSV](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim2/geometry_cases.csv),
  [economy CSV](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim2/economy_cases.csv)
- [Negative controls](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim2/negative_controls.csv)
  and [per-claim nonzero mutation exit](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verifier_failure_controls.json)
- [Independent checker output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json)
- [Method](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim2/method.md)
  and [source audit](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim2/source_audit.md)

Fixed command: `uv run --frozen python repro/src/run_campaign.py` from
`current/`. Run SHA `{{GIT_SHA}}`; CPU/runtime/seeds are on the
[current verification page](#/index).

**Limitation.** The economy conclusion is conditional on the paper's stated
geometric-ladder representation cost. It does not prove that this is the only
reasonable cost model.
