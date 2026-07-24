# Current verification — supersedes the historical rejected baseline

This is the canonical evaluator entrypoint. It supersedes the verifier at
judged Space revision `f457f54c89151cc850279e904d28956e4c23508b`.
The older pages remain unchanged under **Historical rejected baseline**; they
are not the current verification.

## Evidence status

| Claim | Current verdict | Decisive evidence |
| --- | --- | --- |
| [1](#/current-claim-1) | VERIFIED | Dimension-independent symbolic derivation, numerical certificates, official-code parity |
| [2](#/current-claim-2) | VERIFIED | Symbolic regular-simplex and economy derivation, exact-structure sweeps |
| [3](#/current-claim-3) | BLOCKED | Four distinct routes; no released ImageNet checkpoint or predictions |
| [4](#/current-claim-4) | BLOCKED | Four distinct routes; fixed checkpoints and exact rotation transform absent |
| [5](#/current-claim-5) | FALSIFIED | Released 85.97-mIoU entrypoints execute ShapeNetPart, not ModelNet40 |
| [6](#/current-claim-6) | FALSIFIED | Two dynamic CPU counters execute exact 224×224 models and contradict the no-additional-attention-cost statement |

## Reproduce everything

From the downloadable [`current/` source bundle](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/tree/main/current):

```bash
uv sync --frozen
uv run --frozen python repro/src/run_campaign.py
```

The formal fixed command is the second line and is unchanged across experiment
nodes. The first line materializes the single repository-level `.venv` from
the pinned [`pyproject.toml`](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/pyproject.toml),
[`uv.lock`](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/uv.lock),
and [`.python-version`](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.python-version).

Run revision: `{{GIT_SHA}}`  
Lock SHA-256: `{{UV_LOCK_SHA256}}`  
Official code revision: `f2cae70760806451f5e58be4b7e3dc4d0d856a1e`  
Deterministic seeds: Claim 1 `0..31`, Fourier `10000..10063`; Claim 2
`20260719`; Claim 6 `20260724`  
CPU: `{{PLATFORM}}`, `{{CPU_COUNT}}` logical CPUs  
Total formal runtime: `{{TOTAL_RUNTIME_SECONDS}}` seconds

The [cumulative verifier source](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/repro/src/verify_results.py)
reports `{{VERIFIER_CHECK_COUNT}}` checks. Its
[raw output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json)
must contain `"all_checks_pass": true`. The
[mutation-control source](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/repro/src/run_verifier_failure_controls.py)
corrupts one decisive item for each claim; all six runs must exit nonzero, as
recorded in the
[raw failure-control output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verifier_failure_controls.json).
The complete [run metadata](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/run_metadata.json)
contains step runtimes, seeds, CPU details, command, Git SHA, and lock hash.

## Evaluator visibility

Every current claim page shows the exact statement, assumptions, numerical
results, limitations, source links, raw downloads, independent checker, and
negative controls inline. The [visibility matrix](#/current-visibility)
maps those items. Claims 3 and 4 remain BLOCKED: missing evidence is never
converted into a pass. No full-scale ImageNet result was approximated with a
toy or untrained model.
