# Claim 6 — current verification

**Verdict: FALSIFIED.** Claim 6 is conjunctive. The decisive current route
targets Appendix D.4/Table 8's exact statement: for image and video
Transformers, nD-RoPE “only modifies the frequency construction” **without
additional attention cost**, with the parameter increase caused by additional
frequency parameters and negligible overall overhead.

Paper source: [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2606.12146),
retrieved 2026-07-23, SHA-256
`aa7acd683cbb804762be15b607c1862b7be3fd0acd44424c939e2da9f0f6756f`;
anchors Appendix D.4 paragraphs 1 and 3 and Table 8.

## Exact executed-model counterexample

The official released 224×224 baseline uses width `384`; the nD model uses
width `396`. Both exact constructors were executed on CPU under
`torch.inference_mode`. A PyTorch operator profiler and an independent
dispatch-level `FlopCounterMode` measured real forward passes. A separate
closed-form checker attributes MACs by patch projection, QKV, attention
projection, QK, AV, MLP, and classifier.

| Exact model/control | Profiler GMAC | Dispatch GMAC |
| --- | ---: | ---: |
| Official baseline, width 384 | `4.600286208` | `4.598882304` |
| Official nD-RoPE, width 396 | `4.8793826215` | `4.877486784` |
| Matched-width baseline control, width 396 | `4.878849888` | `4.877402112` |
| Monotonic negative control, width 408 | `5.165583552` | `5.164091904` |

Observed nD increases are `6.0669359%` (profiler), `6.0580911%`
(dispatch), and `6.05624997%` (symbolic). The symbolic attention-only increase
is `99,685,152` MACs (`5.6897668%`). The matched-width control explains
`99.8091%` of the profiler delta and `99.9696%` of the dispatch delta.
Live model enumeration finds **zero trainable frequency parameters** and
`432` registered frequency-buffer elements.

This is non-circular: resource levels were not selected from the claimed
formula. Two counters independently execute the exact models; the symbolic
formula is an attribution checker, not the measurement used to establish
scaling.

## Assumptions, controls, and retained ablations

The counterexample uses the released image constructors, batch 1,
`3×224×224` input, finite outputs, and the paper's MAC convention. Required
controls check input resolution, output shape, dual-counter agreement,
Table-8 tolerance, width attribution, monotonic width scaling, and zero
trainable frequency parameters.

The older Table 6/7 arithmetic is retained only as corroboration. In particular,
Table 7's own 2,048-point training-grid row reports `85.80` for `θ=2` and
`85.58` for `θ=100`, but the executed Table 8 route above is the current
decisive verifier.

## Inspect and rerun

- [Exact claim contract](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim6/claim_contract.json)
- [Executable dynamic profiler](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/repro/src/run_claim6_flops.py)
- [Raw dynamic report](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim6/dynamic_flop_report.json)
  and [raw dynamic controls](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim6/dynamic_flop_negative_controls.json)
- [Independent checker output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json)
  and [per-claim nonzero mutation exit](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verifier_failure_controls.json)
- [Method](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim6/method.md),
  [source audit](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim6/source_audit.md),
  and [official nD constructor](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/vendor/nD-RoPE/rope-vit-ndrope/deit/models_v2_ndRope.py)

Fixed command: `uv run --frozen python repro/src/run_campaign.py` from
`current/`. Seed `20260724`; run SHA `{{GIT_SHA}}`; CPU/runtime are on the
[current verification page](#/index).

**Limitations and deviations.** Table 6/7 trained accuracies were not
regenerated because configs/checkpoints are absent. The falsification is
instead the exact Appendix D.4 executed-architecture subclaim. It does not
claim that a 6% operation increase necessarily causes lower accuracy.
