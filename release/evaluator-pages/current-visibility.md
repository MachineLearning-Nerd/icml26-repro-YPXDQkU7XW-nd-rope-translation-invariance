# Evaluator-visible evidence matrix

This matrix is for the current verification at run SHA `{{GIT_SHA}}`. All
links are reachable from the canonical page without internal repository or
dashboard knowledge.

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [Current Claim 1](#/current-claim-1) | Yes | Yes | [CSV/JSON](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/tree/main/current/outputs/claim1) | [Output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json) | Additive + mutation | Yes; universal symbolic derivation | Discoverable |
| 2 | [Current Claim 2](#/current-claim-2) | Yes | Yes | [CSV/JSON](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/tree/main/current/outputs/claim2) | [Output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json) | Malformed simplex + mutation | Yes; universal symbolic derivation | Discoverable |
| 3 | [Current Claim 3](#/current-claim-3) | Yes | Yes | [JSON](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim34/claim34_report.json) | [Output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json) | Evidence-gate + mutation | Yes; full ImageNet metric | BLOCKED, discoverable |
| 4 | [Current Claim 4](#/current-claim-4) | Yes | Yes | [JSON](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim34/claim34_report.json) | [Output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json) | Evidence-gate + mutation | Yes; fixed-model 30° protocol | BLOCKED, discoverable |
| 5 | [Current Claim 5](#/current-claim-5) | Yes | Yes | [JSON](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/source_audit/source_audit.json) | [Output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json) | Entrypoint-use + mutation | Yes; benchmark identity conjunct | FALSIFIED, discoverable |
| 6 | [Current Claim 6](#/current-claim-6) | Yes | Yes | [JSON](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/claim6/dynamic_flop_report.json) | [Output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json) | Matched width, monotonic width, mutation | Yes; exact Appendix D.4 cost statement | FALSIFIED, discoverable |

The fixed command, environment, revision, seeds, CPU, and runtime are shown on
the [canonical current page](#/index). The six-case
[failure-control output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verifier_failure_controls.json)
records a nonzero verifier exit for each deliberately corrupted claim.

