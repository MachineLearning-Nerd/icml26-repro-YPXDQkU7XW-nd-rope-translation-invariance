# Branch audit

The original repository used `master` and opaque `orx/*` experiment branches.
Each branch is retained as a descriptive final branch; no experiment content
is silently merged into `main`.

| Original branch | Original tip | Final branch | Final GitHub tip | Purpose |
| --- | --- | --- | --- | --- |
| `master` | `b9077dbb48340d844aac5c34a456eabe717bdbe1` | `main` | current tip; see the GitHub branches API | Current publication surface. |
| `orx/baseline-judged-6-12-evidence` | `7eb190b46cb1a360ab3db71c927ff30575d95b06` | `baseline/judged-6-of-12` | `e85521dbd435cd45da5de2ca3dd7a89c9959f508` | Frozen judged evidence baseline. |
| `orx/claim-6-exact-table-contradiction` | `a3377365c2d963928d95a36e5492c8fd077bec2a` | `audit/claim-6-table-contradiction` | `1669fd028d17f85b0cec36ac7a875e5076cac935` | Exact C6 table-contract contradiction. |
| `orx/claims-3-4-route-1-artifact-provenance` | `3551401b9142af0c722dab9c5e35091d0f300cb9` | `audit/claims-3-4-artifact-provenance` | `bdb64e2ff9a143d74ab1010bc0abfc187339c9bd` | Checkpoint and prediction provenance. |
| `orx/claims-3-4-route-2-architecture-contract` | `8ae5f8062ec226aa8e41609c4333f3653ec48957` | `audit/claims-3-4-architecture-contract` | `ec66dfd0f974f6be177754295b52fa2ecfbd139d` | Backbone and training-contract audit. |
| `orx/claims-3-4-route-3-cross-table-protocol` | `c2f58c8f07c50896991ba29d17043f0cbb0aafe0` | `audit/claims-3-4-cross-table` | `9ddf41aa2c341971e2f1d3da4c9642ff7bdef47b` | Cross-table/protocol diagnostic. |
| `orx/claims-3-4-route-4-falsification-search` | `3f9f12c83cf273fc257c1931cb50ca8f65de119e` | `audit/claims-3-4-falsification` | `8c4ffd1ed2b1c14638e9fdbebdd60b8fbe82425d` | Assumption-preserving falsification search. |
| `orx/release-candidate-evidence-and-report` | `e9ee75c344ba013d37d8ec7cd585c1e69f21ee73` | `release/cumulative-evidence` | `7b47bbb412cc690129758e49e88812170733a145` | Cumulative claim evidence and report. |
| `orx/final-approval-candidate` | `c1d4e3c0c062d4ff3d142a444dade0f91e12a1b4` | `release/final-approval-candidate` | `0e994b3978d84a495e98daac7a7719b8d1904e29` | Frozen pre-publication candidate. |
| `orx/post-judge-c6-dynamic-flop-attribution` | `2bc6e70f4cb38a04e5f3fee18db3855234435232` | `audit/post-judge-c6-flops` | `dc8fdb922269f86b42d621f3b3a87c2b17c2bfb8` | Exact-model dynamic FLOP/parameter attribution. |
| `orx/post-judge-c6-dual-profiler-release-candidate` | `4723db2062ec0e67b5a857b79c6b8425ac0c33cf` | `release/post-judge-c6-dual-profiler` | `0e707050bf26152ef37bef798f93b529cd62f76c` | Independent dual-counter release package. |
| `orx/post-judge-final-publication-package` | `74554baaba3975ba6104c854a1c3661b5541b14c` | `release/post-judge-publication` | `ac1c7cfd6e82009f134045f256bdec2259457256` | Immutable post-judge publication package. |
| `orx/evaluator-visible-current-verification` | `c3283858f04836356ce806a7b884a228c5cfb954` | `release/evaluator-visible-verification` | `67341358980f9092726d354829fe2db65794748c` | Canonical current evaluator-visible bundle. |
| `orx/blind-review-candidate-package` | `f38ab2213b1f5f27caed0117326c1f5f3362be70` | `release/blind-review-candidate` | `672ea68603dd524879b3e5568d69e47bf3da8b08` | Independently traversed blind-review candidate. |

The final GitHub branch tips above were read from the GitHub branches API after
the rename. Every reachable approved commit is
rewritten to the exact identity `MachineLearning-Nerd
<MachineLearning-Nerd@users.noreply.github.com>`.
