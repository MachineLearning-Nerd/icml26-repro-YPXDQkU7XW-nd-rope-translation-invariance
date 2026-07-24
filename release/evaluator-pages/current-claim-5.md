# Claim 5 — current verification

**Verdict: FALSIFIED as written.** The conjunctive claim says nD-RoPE reaches
`75.85%` top-1 on Kinetics-400, `85.97%` instance-average mIoU on
**ModelNet40 point clouds**, and `71.91%` mIoU on SemanticKITTI. One false
required conjunct falsifies the claim as written.

Paper source: [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2606.12146),
retrieved 2026-07-23, SHA-256
`aa7acd683cbb804762be15b607c1862b7be3fd0acd44424c939e2da9f0f6756f`;
anchors Tables 2–4 and Section 5.1.

## Assumption-satisfying source counterexample

At official commit `f2cae70760806451f5e58be4b7e3dc4d0d856a1e`,
the two released executable entrypoints associated with the `85.97` mIoU
result:

```text
load shapenetcore_partanno_segmentation_benchmark_v0_normal
construct PartNormalDataset
use 16 object categories and 50 part labels
compute instance-average part IoU
never call ModelNetDataLoader.
```

The separate ModelNet40 path is 40-way classification, not part segmentation.
The complete 154-file, manifest-verified release also contains zero
SemanticKITTI files. The decisive contradiction is benchmark identity in the
authors' own entrypoints; it does not rely on a failed training run.

| Numerical/source audit | Result |
| --- | ---: |
| Official files / manifest entries | `154 / 154` |
| Checkpoints | `0` |
| SemanticKITTI files | `0` |
| Shape categories | `16` |
| Part labels | `50` |
| Entrypoints calling ModelNet loader | `false` |

## Inspect and rerun

- [Exact claim contract](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim5/claim_contract.json)
- [Executable source auditor](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/repro/src/run_source_audit.py)
- [Raw audit JSON](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/source_audit/source_audit.json)
  and [154-file inventory](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/source_audit/official_file_inventory.json)
- [Pinned training entrypoint](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/vendor/nD-RoPE/PCT/Point-Transformers-ndrope-vector/train_partseg_ndrope.py)
  and [evaluation entrypoint](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/vendor/nD-RoPE/PCT/Point-Transformers-ndrope-vector/test_partseg.py)
- [Independent checker output](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verification.json)
  and [per-claim nonzero mutation exit](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/outputs/verifier_failure_controls.json)
- [Negative-control method](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim5/method.md)
  and [source audit narrative](https://huggingface.co/spaces/DineshAI/YPXDQkU7XW/resolve/main/current/.openresearch/artifacts/claim5/source_audit.md)

The negative control confirms that mere `ModelNetDataLoader` text in a shared
dataset module is insufficient; only executable entrypoint use counts.
Fixed command: `uv run --frozen python repro/src/run_campaign.py` from
`current/`. Run SHA `{{GIT_SHA}}`.

**Limitation.** Kinetics-400 and SemanticKITTI numeric results are not
individually declared false; the verdict is for the exact conjunctive claim.
