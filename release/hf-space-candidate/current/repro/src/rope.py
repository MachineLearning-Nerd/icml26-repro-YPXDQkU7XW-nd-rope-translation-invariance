"""nD-RoPE translation-invariant formulation (ICML 2026, YPXDQkU7XW).
RoPE applies rotation R_{mθ} to position m. The attention inner product:
  (R_{mθ} q)^T (R_{nθ} k) = q^T R_{mθ}^T R_{nθ} k = q^T R_{(n-m)θ} k
depends ONLY on relative position (n-m). This is translation-invariance."""
import numpy as np

def rope_matrix(m, theta=1.0, d=2):
    """2D rotation by angle mθ (applied per pair of dims)."""
    a = m * theta
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def rope_attention(q, k, m, n, theta=1.0):
    """(R_m q)^T (R_n k) -- the RoPE-modified attention inner product."""
    Rm, Rn = rope_matrix(m, theta), rope_matrix(n, theta)
    return float((Rm @ q) @ (Rn @ k))

def relative_attention(q, k, delta, theta=1.0):
    """q^T R_δ k -- should equal rope_attention(q,k,m,n) for δ=n-m."""
    return float(q @ (rope_matrix(delta, theta) @ k))

def additive_pe_attention(q, k, pm, pn):
    """Negative control: (q+pm)^T (k+pn) -- NOT translation-invariant (depends on absolute m,n)."""
    return float((q + pm) @ (k + pn))
