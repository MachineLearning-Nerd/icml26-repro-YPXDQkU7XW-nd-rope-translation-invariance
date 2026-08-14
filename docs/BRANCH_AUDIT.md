# Branch audit

The original repository used `master` and opaque `orx/*` experiment branches.
Each branch is retained as a descriptive final branch; no experiment content
is silently merged into `main`.

| Original branch | Original tip | Final branch | Purpose |
| --- | --- | --- | --- |
| `master` | `b9077dbb48340d844aac5c34a456eabe717bdbe1` | `main` | Current publication surface. |
| `orx/baseline-judged-6-12-evidence` | `7eb190b46cb1a360ab3db71c927ff30575d95b06` | `baseline/judged-6-of-12` | Frozen judged evidence baseline. |
| `orx/claim-6-exact-table-contradiction` | `a3377365c2d963928d95a36e5492c8fd077bec2a` | `audit/claim-6-table-contradiction` | Exact C6 table-contract contradiction. |
| `orx/claims-3-4-route-1-artifact-provenance` | `3551401b9142af0c722dab9c5e35091d0f300cb9` | `audit/claims-3-4-artifact-provenance` | Checkpoint and prediction provenance. |
| `orx/claims-3-4-route-2-architecture-contract` | `8ae5f8062ec226aa8e41609c4333f3653ec48957` | `audit/claims-3-4-architecture-contract` | Backbone and training-contract audit. |
| `orx/claims-3-4-route-3-cross-table-protocol` | `c2f58c8f07c50896991ba29d17043f0cbb0aafe0` | `audit/claims-3-4-cross-table` | Cross-table/protocol diagnostic. |
| `orx/claims-3-4-route-4-falsification-search` | `3f9f12c83cf273fc257c1931cb50ca8f65de119e` | `audit/claims-3-4-falsification` | Assumption-preserving falsification search. |
| `orx/release-candidate-evidence-and-report` | `e9ee75c344ba013d37d8ec7cd585c1e69f21ee73` | `release/cumulative-evidence` | Cumulative claim evidence and report. |
| `orx/final-approval-candidate` | `c1d4e3c0c062d4ff3d142a444dade0f91e12a1b4` | `release/final-approval-candidate` | Frozen pre-publication candidate. |
| `orx/post-judge-c6-dynamic-flop-attribution` | `2bc6e70f4cb38a04e5f3fee18db3855234435232` | `audit/post-judge-c6-flops` | Exact-model dynamic FLOP/parameter attribution. |
| `orx/post-judge-c6-dual-profiler-release-candidate` | `4723db2062ec0e67b5a857b79c6b8425ac0c33cf` | `release/post-judge-c6-dual-profiler` | Independent dual-counter release package. |
| `orx/post-judge-final-publication-package` | `74554baaba3975ba6104c854a1c3661b5541b14c` | `release/post-judge-publication` | Immutable post-judge publication package. |
| `orx/evaluator-visible-current-verification` | `c3283858f04836356ce806a7b884a228c5cfb954` | `release/evaluator-visible-verification` | Canonical current evaluator-visible bundle. |
| `orx/blind-review-candidate-package` | `f38ab2213b1f5f27caed0117326c1f5f3362be70` | `release/blind-review-candidate` | Independently traversed blind-review candidate. |

The final GitHub branch tips are recorded in the publication gate output and
verified again after the repository rename. Every reachable approved commit is
rewritten to the exact identity `MachineLearning-Nerd
<MachineLearning-Nerd@users.noreply.github.com>`.
