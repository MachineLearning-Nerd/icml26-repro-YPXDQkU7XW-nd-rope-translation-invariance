# Repro — nD-RoPE Translation Invariance (YPXDQkU7XW)

C1: "nD-RoPE provides a unified theoretical formulation for rotary embeddings based on translation-invariant formulation in Hilbert space." RoPE's rotation gives translation-invariant attention: `(R_m q)^T(R_n k) = q^T R_{n-m} k` (depends only on relative position).

| Claim | Verdict | Evidence |
|---|---|---|
| **C1** translation-invariant formulation | **VERIFIED** | max err <1e-10 over 10 instances; additive-PE negative control breaks 5+/10. |

2/2 tests. C2 (performance gains) = empirical, out of scope.
