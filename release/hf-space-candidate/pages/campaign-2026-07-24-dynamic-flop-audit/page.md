# Post-judge Claim 6 dynamic FLOP audit

## Why this route was added

The 2026-07-23 judge correctly treated the earlier Table 6–8 arithmetic as an
internal-consistency check rather than experimental falsification. This route
therefore executes the exact released 224×224 image architectures on CPU.

## Exact claim contract

Appendix D.4 says nD-RoPE for image and video tasks only modifies frequency
construction “without introducing additional attention cost,” and attributes
the parameter increase to additional frequency parameters. The released
baseline has width 384, while the released nD-RoPE constructor has width 396.

The claim is falsified only if:

1. two independent dynamic operation counters detect a strict increase;
2. a separate symbolic checker localizes a strict attention-cost increase;
3. a matched-width non-RoPE control attributes the dynamic delta to width; and
4. live model enumeration finds no trainable frequency parameters.

Every condition is fail-closed in the fixed campaign.

## Executed evidence

With one deterministic 224×224 float32 input:

| Executed architecture | `torch.profiler` GMAC | Parameters |
| --- | ---: | ---: |
| released DeiT-S, width 384 | 4.6003 | 22.0595M |
| released nD-RoPE, width 396 | 4.8794 | 23.3555M |
| matched-width baseline control, width 396 | 4.8788 | 23.4331M |

The profiler increase is 6.07%, matching both the paper-scale total and the
independent width formula. The matched-width control reproduces 99.81% of the
official model delta. The symbolic breakdown finds 99.69 million additional
attention MACs, or 5.69% more attention computation. The live nD-RoPE model
contains zero trainable frequency parameters; its 432 frequency-direction
values are registered buffers.

The second dynamic route uses PyTorch `FlopCounterMode`, which intercepts
supported operations at dispatch level and does not consume profiler events
or the symbolic formulas. The raw JSON records its exact total and the
cross-counter agreement checks.

## Negative controls and scope

A width-408 model must exceed the width-396 model under both dynamic counters
and in parameter count. The width-396 non-RoPE model must be closer to nD-RoPE
than the width-384 baseline. All controls must pass or the campaign exits
nonzero.

This route does not regenerate the unavailable Table 6/7 trained accuracy
ablations. Its verdict is scoped to the exact Appendix D.4 computational-cost
and parameter-cause statements. It does not depend on whether a 6.07% increase
is subjectively “negligible.”

## Durable evidence

- `evidence/2026-07-24/dynamic_flop_report.json`
- `evidence/2026-07-24/dynamic_flop_negative_controls.json`
- `evidence/2026-07-24/verification.json`
- `evidence/2026-07-24/run_metadata.json`
- `evidence/2026-07-24/EVAL.md`

All files from the exact previously judged revision
`f457f54c89151cc850279e904d28956e4c23508b` remain present. Existing evidence
pages are byte-identical.
