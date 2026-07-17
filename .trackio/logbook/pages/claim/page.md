# Claim 1 — Translation invariance
RoPE attention (R_m q)^T(R_n k) = q^T R_{n-m} k depends only on relative position. Verified max err <1e-10. Additive-PE control breaks (not invariant).
