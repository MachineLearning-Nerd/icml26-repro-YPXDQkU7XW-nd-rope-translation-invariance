# Evaluator-blind review — exact candidate

Review date: 2026-07-24  
Candidate Git revision: `f38ab2213b1f5f27caed0117326c1f5f3362be70`  
Downloaded GitHub archive SHA-256:
`fa1a0eccd92d4fb7fcac3f6d031aa7e10f1701cbb03b5277b247bba59cfec852`  
Candidate files: 309  
Review result: **PASS**

The candidate revision was downloaded into the empty directory
`/tmp/ndrope-final-blind.oBfmQS`. The reviewer was given only the downloaded
candidate and the judge/evaluator rubric. Review began at `README.md` and
`logbook.json`; no orx log, dashboard artifact, unpublished branch, or
unlinked internal path was used.

The logbook default resolves to
`current/pages/current-verification.md`. Its first six navigation children are
the six current claim pages. All older page entries are explicitly titled
“Historical rejected baseline,” and the historical page states which current
code and run revision supersede them.

## Visibility matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `current/pages/current-claim-1.md` | Yes | Yes | Yes | 52-check output | Additive and mutation controls | Yes; universal symbolic Fourier/rotation derivation | VERIFIED evidence discoverable |
| 2 | `current/pages/current-claim-2.md` | Yes | Yes | Yes | 52-check output | Malformed-simplex and mutation controls | Yes; universal simplex/economy derivation | VERIFIED evidence discoverable |
| 3 | `current/pages/current-claim-3.md` | Yes | Yes | Yes | 52-check output | Evidence-gate and mutation controls | Yes; full 50,000-image metric | BLOCKED evidence discoverable |
| 4 | `current/pages/current-claim-4.md` | Yes | Yes | Yes | 52-check output | Evidence-gate and mutation controls | Yes; fixed-model 30-degree protocol | BLOCKED evidence discoverable |
| 5 | `current/pages/current-claim-5.md` | Yes | Yes | Yes | 52-check output | Entrypoint-use and mutation controls | Yes; benchmark-identity conjunct | FALSIFIED evidence discoverable |
| 6 | `current/pages/current-claim-6.md` | Yes | Yes | Yes | 52-check output | Matched-width, monotonic-width, and mutation controls | Yes; Appendix D.4 additional-attention-cost statement | FALSIFIED evidence discoverable |

## Checks performed

- All 54 distinct candidate-local link targets exposed by the current pages
  exist.
- The reviewer opened 62 files, including all current pages and every linked
  claim contract, method, source audit, executable, raw result, checker, and
  control needed to reach a verdict.
- The cumulative verifier contains 52 checks; every check is true.
- Six claim-specific evidence mutations each produce exit code 1 and the
  expected failed check.
- Claim 1’s 7,680 rotary and 192 Parseval trials and displayed error values
  match raw JSON/CSV. The page makes the all-real-phase symbolic derivation
  decisive and labels finite trials corroboration.
- Claim 2’s 7,936 symmetry trials, 64 optimizations, and error bounds match raw
  data. The page displays the all-`n` simplex and economy derivation.
- Claims 3 and 4 remain BLOCKED after routes 1–4; no missing checkpoint,
  untrained model, subset, or table transcription is promoted.
- Claim 5’s displayed 154-file inventory, 16 categories, 50 part labels, and
  absent executable ModelNet loader use match the source-audit JSON.
- Claim 6’s profiler (`6.066935857%`), dispatch (`6.058091110%`), attention
  (`5.689766839%`), zero trainable frequency parameters, and 432 buffer
  elements match raw output.
- The fixed command, `uv` environment, lock hash, run SHA, CPU, runtime, and
  corrected seed ranges (`0..31`, `10000..10063`) match run metadata.

## Files opened

1. `README.md`
2. `logbook.json`
3. `current/pages/current-verification.md`
4. `current/pages/current-claim-1.md`
5. `current/pages/current-claim-2.md`
6. `current/pages/current-claim-3.md`
7. `current/pages/current-claim-4.md`
8. `current/pages/current-claim-5.md`
9. `current/pages/current-claim-6.md`
10. `current/pages/current-visibility.md`
11. `current/pages/historical-rejected-baseline.md`
12. `current/pyproject.toml`
13. `current/uv.lock`
14. `current/.python-version`
15. `current/repro/src/verify_results.py`
16. `current/outputs/verification.json`
17. `current/repro/src/run_verifier_failure_controls.py`
18. `current/outputs/verifier_failure_controls.json`
19. `current/.openresearch/artifacts/run_metadata.json`
20. `current/repro/src/run_symbolic_proofs.py`
21. `current/outputs/symbolic_proof_certificates.json`
22. `current/.openresearch/artifacts/claim1/claim_contract.json`
23. `current/repro/src/run_claim1.py`
24. `current/outputs/claim1/claim1_report.json`
25. `current/outputs/claim1/translation_trials.csv`
26. `current/outputs/claim1/parseval_trials.json`
27. `current/.openresearch/artifacts/claim1/method.md`
28. `current/.openresearch/artifacts/claim1/source_audit.md`
29. `current/.openresearch/artifacts/claim2/claim_contract.json`
30. `current/repro/src/run_claim2.py`
31. `current/outputs/claim2/claim2_report.json`
32. `current/outputs/claim2/geometry_cases.csv`
33. `current/outputs/claim2/economy_cases.csv`
34. `current/outputs/claim2/negative_controls.csv`
35. `current/.openresearch/artifacts/claim2/method.md`
36. `current/.openresearch/artifacts/claim2/source_audit.md`
37. `current/.openresearch/artifacts/claim3/claim_contract.json`
38. `current/repro/src/run_claim34.py`
39. `current/outputs/claim34/claim34_report.json`
40. `current/outputs/claim34/negative_controls.json`
41. `current/.openresearch/artifacts/claim3/attempts.md`
42. `current/.openresearch/artifacts/claim3/source_audit.md`
43. `current/.openresearch/artifacts/claim3/method.md`
44. `current/.openresearch/artifacts/claim4/claim_contract.json`
45. `current/.openresearch/artifacts/claim4/attempts.md`
46. `current/.openresearch/artifacts/claim4/source_audit.md`
47. `current/.openresearch/artifacts/claim4/method.md`
48. `current/.openresearch/artifacts/claim5/claim_contract.json`
49. `current/repro/src/run_source_audit.py`
50. `current/outputs/source_audit/source_audit.json`
51. `current/outputs/source_audit/official_file_inventory.json`
52. `current/vendor/nD-RoPE/PCT/Point-Transformers-ndrope-vector/train_partseg_ndrope.py`
53. `current/vendor/nD-RoPE/PCT/Point-Transformers-ndrope-vector/test_partseg.py`
54. `current/.openresearch/artifacts/claim5/method.md`
55. `current/.openresearch/artifacts/claim5/source_audit.md`
56. `current/.openresearch/artifacts/claim6/claim_contract.json`
57. `current/repro/src/run_claim6_flops.py`
58. `current/outputs/claim6/dynamic_flop_report.json`
59. `current/outputs/claim6/dynamic_flop_negative_controls.json`
60. `current/.openresearch/artifacts/claim6/method.md`
61. `current/.openresearch/artifacts/claim6/source_audit.md`
62. `current/vendor/nD-RoPE/rope-vit-ndrope/deit/models_v2_ndRope.py`

No navigation or scientific gap was found in this traversal. This result is a
pre-publication candidate audit, not a live judge score.
