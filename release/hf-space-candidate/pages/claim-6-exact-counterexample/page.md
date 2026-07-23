# Claim 6: exact counterexample

**Verdict: FALSIFIED. Confidence: HIGH.**

Appendix D.3 says moderate bases such as θ=100 “consistently achieve the best
performance across all settings” and describes θ=100 as strongest across all
point densities. At the stated 2,048-point training grid, Table 7 reports:

| Input points | θ=2 | θ=100 | Strict margin |
| ---: | ---: | ---: | ---: |
| 2,048 | **85.80** | 85.58 | **+0.22 percentage points** |

The counterexample preserves the paper’s model family, candidate bases,
instance-average mIoU metric, and explicit training-grid condition. It
therefore contradicts the universal subclaim within its stated domain.

## Controls and limitations

Removing the universal quantifier or averaging across all rows supports
θ=100; both controls pass. This confirms that the verdict depends on the exact
statement rather than a nearby stronger interpretation.

No ablation checkpoint is released, so the trained values were not
independently regenerated. The falsification is an internal consistency check
between the paper’s exact prose and Table 7, not a failed training
reproduction. The verdict does not depend on the subjective meaning of
“negligible” FLOPs.

Machine-readable evidence and controls are in
`evidence/2026-07-23/claim6_report.json` and
`evidence/2026-07-23/claim6_negative_controls.json`.
