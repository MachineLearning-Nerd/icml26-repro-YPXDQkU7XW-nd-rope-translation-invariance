# Evaluator-blind review: failed 74554ba candidate

Candidate Git revision:
`74554baaba3975ba6104c854a1c3661b5541b14c`.

Review date: 2026-07-24.

The candidate was downloaded from GitHub into an empty temporary directory.
The review began at `README.md`, `logbook.json`, and `pages/index.md`. No
internal repository files, experiment logs, dashboard artifacts, or
unpublished branches were consulted while scoring visibility.

## Files opened

1. `README.md`
2. `logbook.json`
3. `pages/index.md`
4. `pages/executive-summary/page.md`
5. `pages/claim-1-fourier-hilbert-derivation/page.md`
6. `pages/claim-2-simplex-geometry-and-optimal-scale/page.md`
7. `pages/claim-3-imagenet-in-domain-accuracy/page.md`
8. `pages/claim-4-30-degree-rotation-robustness/page.md`
9. `pages/claim-5-cross-modal-benchmarks/page.md`
10. `pages/claim-6-ablations-and-computational-cost/page.md`
11. `pages/conclusion/page.md`
12. `pages/campaign-2026-07-23-summary/page.md`
13. `pages/claim-3-four-route-audit/page.md`
14. `pages/claim-4-four-route-audit/page.md`
15. `pages/claim-6-exact-counterexample/page.md`
16. `pages/campaign-2026-07-24-dynamic-flop-audit/page.md`

## Initial visibility matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | historical Claim 1 page | partial historical code | yes | historical bucket link | rejected 19-check checker | yes | quantifier/assumption scope incomplete | NOT RELEASE-READY |
| 2 | historical Claim 2 page | partial historical code | yes | historical bucket links | rejected 19-check checker | yes | universal proof scope not separated from finite trials | NOT RELEASE-READY |
| 3 | two conflicting historical pages | no current code | partial | no clickable current raw link | no current output | no downloadable control | contract stated | NOT RELEASE-READY |
| 4 | two conflicting historical pages | no current code | partial | no clickable current raw link | no current output | no downloadable control | contract stated | NOT RELEASE-READY |
| 5 | historical Claim 5 page | historical source-audit code | yes | no current raw JSON link | rejected 19-check checker | incomplete | conjunctive claim stated | NOT RELEASE-READY |
| 6 | post-judge page appears last | no | yes | paths rendered as code, not links | no current checker output | summarized only | Appendix D.4 contract stated | NOT RELEASE-READY |

## Conclusions the blind reviewer could not verify

- `pages/index.md` does not link the post-judge verifier at all.
- `logbook.json` puts the post-judge page after every historical page.
- The old conclusion still presents the rejected 20-test/19-check verifier as
  the current “Verification run.”
- No page labels the 2026-07-23 verifier as a historical rejected baseline.
- The post-judge page has zero Markdown links. Its raw paths are not
  discoverable downloads.
- The current 44-check verifier source and output are absent from the current
  page.
- The exact reproduction Git SHA, deterministic seed, pinned lock hash, CPU
  runtime, and deviations are not co-located with the current verifier.
- Claims 3 and 4 do not expose current executable route code, raw JSON,
  independent checker output, or negative controls through clickable links.
- No claim shows a mutation test proving the verifier exits nonzero when that
  claim's evidence is corrupted.
- The fixed `uv run --frozen python repro/src/run_campaign.py` command and
  locked environment are stated, but the source/environment needed to inspect
  them are not present in the candidate artifact.
- Claims 1 and 2 rely visibly on finite numerical trials; their universal
  symbolic derivations and assumptions are not presented as the decisive
  evidence.

Overall blind verdict: **PUBLICATION BLOCKED**.
