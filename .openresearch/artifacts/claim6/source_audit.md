# Claim 6 source audit

Paper source: `https://ar5iv.labs.arxiv.org/html/2606.12146`, retrieved
2026-07-23, SHA-256
`aa7acd683cbb804762be15b607c1862b7be3fd0acd44424c939e2da9f0f6756f`.

Appendix D.3 anchor `A4.SS3.SSS0.Px2.p1` quantifies over “all settings” and
says theta=100 consistently gives the best performance. Anchor
`A4.SS3.SSS0.Px2.p2` repeats that theta=100 is strongest across all point
densities. Table 7 anchor `A4.T7` states that all models train with 2,048
points. Its 2,048-point row reports 85.80 for theta=2 and 85.58 for theta=100.

This is a strict 0.22-point counterexample at the stated training grid, within
the paper's own model, metric, candidate set, and table. It does not depend on
missing checkpoints or on a proxy experiment.

The secondary audit finds that Table 6's 6x10 allocation has 60 scale-head
slots while all other rows have 64, despite the text stating fixed 384
dimensions, six dimensions per scale, and constant positional channels.
The released image implementation uses a persistent `freqs` buffer, not extra
learnable frequency parameters; therefore Table 8's causal attribution of its
parameter increase to frequency parameters is also unsupported by the release.
