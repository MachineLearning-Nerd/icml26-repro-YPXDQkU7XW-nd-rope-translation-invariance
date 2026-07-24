---
title: "Reproduction: nD-RoPE: A Generalized RoPE for n-Dimensional Position Embedding"
emoji: 🎯
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
tags:
 - trackio
 - trackio-logbook
 - open-experiment
 - icml2026-repro
 - paper-YPXDQkU7XW
---

# Reproduction: nD-RoPE: A Generalized RoPE for n-Dimensional Position Embedding

## Current evaluator verification

Open the Space logbook at its default page. The default is now **Current
verification**, with one self-contained page per claim, executable source,
inline results, downloadable raw data, independent checks, negative controls,
and six nonzero verifier-mutation exits.

The downloadable [`current/` bundle](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/tree/main/current)
reproduces the evidence with:

```bash
uv sync --frozen
uv run --frozen python repro/src/run_campaign.py
```

Claims 1–2 are VERIFIED with universal symbolic derivations as the decisive
evidence; finite trials are labeled corroboration. Claims 3–4 remain BLOCKED
after four different routes. Claim 5 is FALSIFIED as a conjunctive benchmark
identity statement. Claim 6 is FALSIFIED by two dynamic CPU counters executing
the exact released 224×224 models.

The pages from judged revision
`f457f54c89151cc850279e904d28956e4c23508b` remain reachable and unchanged,
but are explicitly labeled **Historical rejected baseline** and are not the
current verifier.

Published with [Trackio](https://github.com/gradio-app/trackio).
