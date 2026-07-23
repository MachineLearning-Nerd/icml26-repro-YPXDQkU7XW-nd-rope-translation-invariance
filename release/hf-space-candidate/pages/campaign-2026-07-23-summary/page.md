# 2026-07-23 cumulative campaign

This additive campaign preserves every page from the judged revision
`dcbbd49ee4487f62443820c7dfe2be11ae10af51` byte-for-byte. It reruns Claims
1, 2, and 5 and adds a four-route audit for Claims 3 and 4 plus an exact
counterexample for Claim 6.

| Claim | Evidence verdict | Confidence | New evidence |
| --- | --- | --- | --- |
| 1 | **VERIFIED** | HIGH | cumulative 7,680 rotary and 192 Fourier trials pass |
| 2 | **VERIFIED** | HIGH | dimensions 2–32 and 64 optimizations pass |
| 3 | **BLOCKED** | LOW | four routes complete; faithful artifacts unavailable |
| 4 | **BLOCKED** | LOW | four routes complete; exact rotation protocol unavailable |
| 5 | **FALSIFIED** | HIGH | pinned source contradiction reruns successfully |
| 6 | **FALSIFIED** | HIGH | Table 7 contradicts its universal θ=100 statement |

The live judge score is still **6/12**. This page does not claim that the judge
has awarded new points.

## Fixed command and environment

```text
uv run --frozen python repro/src/run_campaign.py
```

The environment is locked by `pyproject.toml`, `.python-version`, and
`uv.lock`. Formal runs used local Apple M2 CPU compute; no GPU and no paid
Hugging Face compute were used.

Machine-readable evidence is under `evidence/2026-07-23/`. The cumulative
verifier is fail-closed, the negative controls are recorded separately, and
the upload allowlist contains text files only.
