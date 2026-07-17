import os, sys; import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import rope
def test_translation_invariance():
    rng = np.random.default_rng(0)
    for _ in range(10):
        q, k = rng.standard_normal(2), rng.standard_normal(2)
        m, n = int(rng.integers(0, 100)), int(rng.integers(0, 100))
        t = rng.uniform(0.1, 2.0)
        assert abs(rope.rope_attention(q, k, m, n, t) - rope.relative_attention(q, k, n-m, t)) < 1e-10
def test_additive_pe_not_invariant():
    rng = np.random.default_rng(1); b = 0
    for _ in range(10):
        q, k = rng.standard_normal(2), rng.standard_normal(2)
        pm, pn = rng.standard_normal(2), rng.standard_normal(2)
        v1 = rope.additive_pe_attention(q, k, pm, pn)
        v2 = rope.additive_pe_attention(q, k, pm+1, pn+1)
        if abs(v1-v2) > 0.01: b += 1
    assert b >= 5
if __name__ == "__main__": import pytest; sys.exit(pytest.main([__file__, "-v"]))
