# Claim 1 — current verification

**Verdict: VERIFIED.** Paper Section 4.1 and Equation 6 claim that, for
arbitrary dimension, query/key positions and a shared translation, the
Fourier-derived rotary attention depends only on relative displacement. The
claim also says this is one n-dimensional construction rather than an
axis-wise decomposition.

Paper source: [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2606.12146),
retrieved 2026-07-23, SHA-256
`aa7acd683cbb804762be15b607c1862b7be3fd0acd44424c939e2da9f0f6756f`;
anchors Section 4.1 and Equation 6.

## Assumptions and full quantifiers

The derivation assumes real position and wave-vector projections, conjugate
query/key Fourier phases, and a common translation. For every real phase
pair `a,b` and translation `t`,

```text
R(a)^T R(b) = R(b-a)
R(a+t)^T R(b+t) = R(b-a)
exp(-i(u+s)) exp(i(v+s)) = exp(i(v-u)).
```

These identities cover arbitrary ambient `n`: `a=ω·x`, `b=ω·y`,
`s=ω·t` for every wave vector `ω`. Thus the symbolic derivation—not a finite
sweep—is the decisive universal evidence. The
[symbolic checker source](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/repro/src/run_symbolic_proofs.py)
uses exact SymPy simplification for the rotation-group and Fourier identities.
Its [proof certificate](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/symbolic_proof_certificates.json)
also rejects the wrong displacement sign.

## Observed corroboration

| Check | Result |
| --- | ---: |
| Rotary trials | 7,680 |
| Finite Fourier/Parseval trials | 192 |
| Maximum shared-translation error | `1.4210854715202004e-14` |
| Maximum relative-identity error | `1.4210854715202004e-14` |
| Maximum Parseval error | `1.1102230246251565e-15` |
| Official phase error | `3.1401849173675503e-16` |
| Official float32 rotation error | `1.28461053794382e-7` |
| Additive-position control minimum detected shift | `7.863305373945195e-6` |

The numerical domain covers dimensions `1,2,3,4,5,8`, random and simplex
frequency families, float64 clean-room arithmetic, and float32 parity against
official commit `f2cae70760806451f5e58be4b7e3dc4d0d856a1e`.
Thresholds are `1e-10` clean-room and `5e-7` official float32.

## Inspect and rerun

- [Claim contract](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim1/claim_contract.json)
- [Executable Claim 1 source](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/repro/src/run_claim1.py)
- [Raw report](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim1/claim1_report.json),
  [translation CSV](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim1/translation_trials.csv),
  [Parseval JSON](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim1/parseval_trials.json)
- [Independent verifier](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/repro/src/verify_results.py)
  and [output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json)
- [Negative-control rows](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim1/translation_trials.csv)
  and [per-claim nonzero mutation exit](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verifier_failure_controls.json)
- [Method](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim1/method.md)
  and [source audit](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim1/source_audit.md)

Fixed command: `uv run --frozen python repro/src/run_campaign.py` from
`current/`. Run SHA `c3283858f04836356ce806a7b884a228c5cfb954`; CPU/runtime/seeds are shown on the
[current verification page](#/index).

**Limitation.** The finite trials are corroboration only. The universal verdict
rests on the independently reconstructed symbolic derivation under the stated
Fourier-phase assumptions; it does not claim that this is the only possible
translation-invariant embedding.
