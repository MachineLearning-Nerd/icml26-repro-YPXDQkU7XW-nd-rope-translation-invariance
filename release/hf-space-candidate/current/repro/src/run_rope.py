import os, sys, json; import numpy as np; sys.path.insert(0, os.path.dirname(__file__))
import rope
rng = np.random.default_rng(0); res = {}
ok = True; max_err = 0.0
for s in range(10):
    q, k = rng.standard_normal(2), rng.standard_normal(2)
    m, n = int(rng.integers(0, 100)), int(rng.integers(0, 100))
    theta = rng.uniform(0.1, 2.0)
    ra = rope.rope_attention(q, k, m, n, theta)
    rr = rope.relative_attention(q, k, n - m, theta)
    err = abs(ra - rr); max_err = max(max_err, err); ok &= err < 1e-10
# negative control: additive PE is NOT translation-invariant
nc_breaks = 0
for s in range(10):
    q, k = rng.standard_normal(2), rng.standard_normal(2)
    pm, pn = rng.standard_normal(2), rng.standard_normal(2)
    pm2, pn2 = pm.copy(), pn.copy()  # same relative but different absolute
    # for additive PE, (q+pm)^T(k+pn) depends on pm, pn (absolute), not just pn-pm
    v1 = rope.additive_pe_attention(q, k, pm, pn)
    v2 = rope.additive_pe_attention(q, k, pm + np.array([1,1]), pn + np.array([1,1]))
    if abs(v1 - v2) > 0.01: nc_breaks += 1
res = dict(translation_invariant=bool(ok), max_err=float(max_err),
           additive_pe_not_invariant=bool(nc_breaks >= 5), nc_breaks=nc_breaks,
           verified=bool(ok and nc_breaks >= 5))
json.dump(res, open(os.path.join(os.path.dirname(__file__),"..","..","outputs","rope_summary.json"),"w"), indent=2)
print(f"RoPE translation-invariance (YPXDQkU7XW): max err {max_err:.2e}: {'VERIFIED' if ok else 'FAIL'}")
print(f"Additive-PE negative control: {nc_breaks}/10 break (not invariant): {nc_breaks >= 5}")
print(f"Overall: {'VERIFIED' if res['verified'] else 'PARTIAL'}")
