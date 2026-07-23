# Claim 5: Cross-modal benchmarks


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ff6c2a532ad5", "created_at": "2026-07-19T17:01:38+00:00", "title": "Claim 5: Cross-modal benchmarks"}
-->
**FALSIFIED AS WRITTEN.** The executable 85.97-mIoU path is ShapeNetPart part segmentation, not ModelNet40.


---
<!-- trackio-cell
{"type": "code", "id": "cell_7902ae06fdc5", "created_at": "2026-07-19T17:13:53+00:00", "title": "Run: env run_source_audit.py (exit 0)", "command": ["env", "PYTHONPATH=repro/src", "python", "repro/src/run_source_audit.py", "--official-root", "vendor/nD-RoPE", "--output-dir", "outputs/source_audit"], "exit_code": 0, "duration_s": 1.416}
-->
````bash
$ env PYTHONPATH=repro/src python repro/src/run_source_audit.py --official-root vendor/nD-RoPE --output-dir outputs/source_audit
````

exit 0 · 1.4s


````python title=run_source_audit.py
"""Audit the released source against anchored empirical claims 3-6.

This does not promote paper-table transcription to reproduction.  It records
what can and cannot be executed from the pinned release and produces decisive
source-level falsification evidence where the released benchmark identity is
incompatible with the paper claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import torch

from official_extract import load_functions


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_inventory(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    root = args.official_root
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    files = file_inventory(root)
    checkpoints = [
        path
        for path in files
        if Path(path).suffix.lower() in {".pth", ".pt", ".ckpt", ".bin", ".safetensors"}
    ]
    semkitti_files = [path for path in files if re.search("semantic.?kitti", path, re.I)]

    image_nd = root / "rope-vit-ndrope/deit/models_v2_ndRope.py"
    image_baseline = root / "rope-vit-ndrope/deit/models_v2_rope.py"
    train_script = root / "rope-vit-ndrope/deit/run_ndrope.bash"
    point_train = root / "PCT/Point-Transformers-ndrope-vector/train_partseg_ndrope.py"
    point_test = root / "PCT/Point-Transformers-ndrope-vector/test_partseg.py"
    point_dataset = root / "PCT/Point-Transformers-ndrope-vector/dataset.py"

    nd_text = image_nd.read_text(encoding="utf-8")
    baseline_text = image_baseline.read_text(encoding="utf-8")
    point_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (point_train, point_test, point_dataset)
    )
    nd_match = re.search(
        r"ndrope_deit_small_patch16_LS.*?embed_dim=(\d+).*?num_heads=(\d+)",
        nd_text,
        re.S,
    )
    baseline_match = re.search(
        r"rope_axial_deit_small_patch16_LS.*?embed_dim=(\d+).*?num_heads=(\d+)",
        baseline_text,
        re.S,
    )
    if not nd_match or not baseline_match:
        raise AssertionError("could not extract image model dimensions")
    nd_width, nd_heads = map(int, nd_match.groups())
    baseline_width, baseline_heads = map(int, baseline_match.groups())

    official = load_functions(
        image_nd,
        ["generate_simplex_vectors_with_projection", "compute_ndrope_cis"],
    )
    simplex = official["generate_simplex_vectors_with_projection"](2)
    frequencies = simplex[None, None, :, :].repeat(nd_heads, 12, 1, 1)
    positions = torch.stack(
        torch.meshgrid(
            torch.arange(14, dtype=torch.float32),
            torch.arange(14, dtype=torch.float32),
            indexing="xy",
        ),
        dim=-1,
    ).reshape(-1, 2)
    paper_grid_started = time.perf_counter()
    paper_grid_phases = official["compute_ndrope_cis"](
        frequencies,
        positions,
        nd_width // nd_heads,
        nd_heads,
        theta=100.0,
    )
    paper_grid_seconds = time.perf_counter() - paper_grid_started
    standard_width_rejected = False
    try:
        official["compute_ndrope_cis"](
            frequencies,
            positions,
            baseline_width // baseline_heads,
            baseline_heads,
            theta=100.0,
        )
    except AssertionError:
        standard_width_rejected = True

    modelnet_mentions = len(re.findall("modelnet40", point_text, re.I))
    shapenet_mentions = len(re.findall("shapenetcore", point_text, re.I))
    has_16_categories = "num_category = 16" in point_text
    has_50_parts = "args.num_class = 50" in point_text
    has_part_iou = "instance_avg_iou" in point_text or "inctance_avg_iou" in point_text
    modelnet_loader_is_separate_classification = (
        "ModelNetDataLoader" in point_text and "train_cls.py" in " ".join(files)
    )
    # The selected part-segmentation files contain the generic dataset class,
    # so distinguish the executable train/test entrypoints from that definition.
    entrypoint_text = point_train.read_text(encoding="utf-8") + point_test.read_text(
        encoding="utf-8"
    )
    entrypoints_use_partnormal = "PartNormalDataset" in entrypoint_text
    entrypoints_use_shapenet = "shapenetcore_partanno" in entrypoint_text
    entrypoints_use_modelnet = "ModelNetDataLoader" in entrypoint_text

    table8 = {
        "image": {"baseline_flops": 4.61, "ndrope_flops": 4.89, "baseline_params": 22.06, "ndrope_params": 23.36},
        "video": {"baseline_flops": 196.05, "ndrope_flops": 196.05, "baseline_params": 121.0, "ndrope_params": 121.0},
        "point_v1_vector": {"baseline_flops": 36.72, "ndrope_flops": 37.49, "baseline_params": 19.40, "ndrope_params": 19.40},
        "point_v2_vector": {"baseline_flops": 50.04, "ndrope_flops": 51.06, "baseline_params": 11.32, "ndrope_params": 11.33},
    }
    for values in table8.values():
        values["flops_increase_percent"] = 100 * (
            values["ndrope_flops"] / values["baseline_flops"] - 1
        )
        values["params_increase_percent"] = 100 * (
            values["ndrope_params"] / values["baseline_params"] - 1
        )

    table6 = {
        "64x1": [67.54, 69.06, 70.99, 71.55, 72.32, 73.56, 76.31, 77.94],
        "32x2": [73.36, 75.56, 77.89, 77.37, 77.20, 79.16, 79.71, 80.16],
        "16x4": [76.87, 78.45, 80.17, 80.40, 80.92, 81.10, 80.20, 78.90],
        "6x10": [77.09, 79.46, 81.07, 81.54, 82.03, 81.62, 80.78, 79.46],
        "4x16": [76.42, 78.96, 80.55, 80.88, 81.74, 81.94, 80.62, 79.22],
        "2x32": [64.18, 66.50, 69.57, 68.64, 71.64, 75.01, 79.16, 80.24],
        "1x64": [58.00, 63.91, 65.93, 64.50, 67.41, 69.74, 73.09, 74.80],
    }
    table6_means = {key: float(np.mean(values)) for key, values in table6.items()}
    table6_column_winners = []
    for column in range(8):
        table6_column_winners.append(max(table6, key=lambda key: table6[key][column]))

    table7 = {
        "2": [53.59, 78.05, 82.20, 85.43, 85.80, 84.78, 82.47],
        "100": [55.04, 78.80, 82.66, 85.61, 85.58, 84.98, 82.88],
        "1e4": [45.94, 74.29, 80.55, 85.17, 83.66, 84.70, 82.77],
        "1e6": [46.16, 78.74, 81.62, 82.67, 83.57, 84.00, 82.09],
    }
    table7_means = {key: float(np.mean(values)) for key, values in table7.items()}

    report = {
        "official_source": {
            "commit": commit,
            "file_count": len(files),
            "checkpoint_count": len(checkpoints),
            "semantic_kitti_file_count": len(semkitti_files),
            "hashes": {
                str(path.relative_to(root)): sha256(path)
                for path in (image_nd, image_baseline, train_script, point_train, point_test, point_dataset)
            },
        },
        "claim3_imagenet": {
            "assessment": "inconclusive_metric__released_source_has_backbone_confound",
            "paper_metric_reproduced": False,
            "ndrope_width": nd_width,
            "ndrope_heads": nd_heads,
            "ndrope_head_dim": nd_width // nd_heads,
            "baseline_width": baseline_width,
            "baseline_heads": baseline_heads,
            "baseline_head_dim": baseline_width // baseline_heads,
            "width_increase_percent": 100 * (nd_width / baseline_width - 1),
            "standard_deit_s_width_rejected_by_official_phase_layout": standard_width_rejected,
            "paper_resolution_phase_shape": list(paper_grid_phases.shape),
            "paper_resolution_phase_seconds_cpu": paper_grid_seconds,
            "reason": "No released checkpoint; nD source uses width 396 while released axial/mixed baselines use 384.",
        },
        "claim4_rotation": {
            "assessment": "inconclusive",
            "paper_metric_reproduced": False,
            "reason": "The zero-shot test requires the unreleased trained ImageNet checkpoints and validation data.",
        },
        "claim5_cross_modal": {
            "assessment": "falsified_as_written",
            "paper_metrics_reproduced": False,
            "modelnet_segmentation_subclaim": False,
            "decisive_reason": "The released 85.97-mIoU path is ShapeNetPart part segmentation, not ModelNet40.",
            "entrypoints_use_partnormal_dataset": entrypoints_use_partnormal,
            "entrypoints_use_shapenet_part_path": entrypoints_use_shapenet,
            "entrypoints_use_modelnet_loader": entrypoints_use_modelnet,
            "shapenet_mentions": shapenet_mentions,
            "modelnet_mentions_in_combined_dataset_module": modelnet_mentions,
            "uses_16_shape_categories": has_16_categories,
            "uses_50_part_labels": has_50_parts,
            "computes_instance_miou": has_part_iou,
            "separate_modelnet_path_is_classification": modelnet_loader_is_separate_classification,
            "semantic_kitti_implementation_released": bool(semkitti_files),
            "logic": "Claim 5 is conjunctive; one false benchmark identity falsifies it as written even though Kinetics and SemanticKITTI metrics remain untested.",
        },
        "claim6_ablation_cost": {
            "assessment": "partially_reproduced_from_arithmetic__empirical_ablations_unrerun",
            "paper_metrics_reproduced": False,
            "table6_mean_accuracy": table6_means,
            "table6_best_mean": max(table6_means, key=table6_means.get),
            "table6_column_winners": table6_column_winners,
            "table6_6x10_product": 60,
            "fixed_positional_channels_stated_in_paper": 384,
            "dimensions_implied_by_6x10_for_2d_simplex": 360,
            "official_released_scale_head_layout": "11 scales x 6 heads at width 396",
            "table7_mean_miou": table7_means,
            "table7_best_mean": max(table7_means, key=table7_means.get),
            "table8_arithmetic": table8,
            "reason": "Table arithmetic supports 6x10 and theta=100 by mean, but checkpoints/configs for those ablations are absent; image FLOPs and parameters rise about 6%, driven by width 396 rather than learned frequency parameters.",
        },
        "reproducibility_blockers": {
            "checkpoints": checkpoints,
            "semantic_kitti_files": semkitti_files,
            "image_data_placeholder": "xxx/dataset/imagenet" in train_script.read_text(encoding="utf-8"),
            "training_hardware_paper": "4 x NVIDIA A100 40GB",
            "paid_remote_compute_authorized": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "official_file_inventory.json").write_text(
        json.dumps(files, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, default=Path("vendor/nD-RoPE"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/source_audit"))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))

````


````output
{
  "official_source": {
    "commit": "f2cae70760806451f5e58be4b7e3dc4d0d856a1e",
    "file_count": 154,
    "checkpoint_count": 0,
    "semantic_kitti_file_count": 0,
    "hashes": {
      "rope-vit-ndrope/deit/models_v2_ndRope.py": "5c53340da1742636c713a83f40eb982275134fd4e471f7da764d09a5e4a9a51a",
      "rope-vit-ndrope/deit/models_v2_rope.py": "61cf1592955b700e613c466412347a7db7553ee0774c627551f22842ea2934d1",
      "rope-vit-ndrope/deit/run_ndrope.bash": "221a24bd78b50631d52e778c95a05680dc9708ec55eb5dfe7409c05d030d05cb",
      "PCT/Point-Transformers-ndrope-vector/train_partseg_ndrope.py": "23e6974d903461c326bf2c519e2a3b79faa17618bb35f4212300c609ada5f400",
      "PCT/Point-Transformers-ndrope-vector/test_partseg.py": "89e047eb16e629fd5d69ff1a9dfb810072a2520b7eeccea327702ef36c867a5e",
      "PCT/Point-Transformers-ndrope-vector/dataset.py": "1cf2229aa3b4e4862b520eae779f66dca398f37c3bd38424ba6037e83091ae67"
    }
  },
  "claim3_imagenet": {
    "assessment": "inconclusive_metric__released_source_has_backbone_confound",
    "paper_metric_reproduced": false,
    "ndrope_width": 396,
    "ndrope_heads": 6,
    "ndrope_head_dim": 66,
    "baseline_width": 384,
    "baseline_heads": 6,
    "baseline_head_dim": 64,
    "width_increase_percent": 3.125,
    "standard_deit_s_width_rejected_by_official_phase_layout": true,
    "paper_resolution_phase_shape": [
      12,
      6,
      196,
      33
    ],
    "paper_resolution_phase_seconds_cpu": 0.002318291997653432,
    "reason": "No released checkpoint; nD source uses width 396 while released axial/mixed baselines use 384."
  },
  "claim4_rotation": {
    "assessment": "inconclusive",
    "paper_metric_reproduced": false,
    "reason": "The zero-shot test requires the unreleased trained ImageNet checkpoints and validation data."
  },
  "claim5_cross_modal": {
    "assessment": "falsified_as_written",
    "paper_metrics_reproduced": false,
    "modelnet_segmentation_subclaim": false,
    "decisive_reason": "The released 85.97-mIoU path is ShapeNetPart part segmentation, not ModelNet40.",
    "entrypoints_use_partnormal_dataset": true,
    "entrypoints_use_shapenet_part_path": true,
    "entrypoints_use_modelnet_loader": false,
    "shapenet_mentions": 3,
    "modelnet_mentions_in_combined_dataset_module": 4,
    "uses_16_shape_categories": true,
    "uses_50_part_labels": true,
    "computes_instance_miou": true,
    "separate_modelnet_path_is_classification": true,
    "semantic_kitti_implementation_released": false,
    "logic": "Claim 5 is conjunctive; one false benchmark identity falsifies it as written even though Kinetics and SemanticKITTI metrics remain untested."
  },
  "claim6_ablation_cost": {
    "assessment": "partially_reproduced_from_arithmetic__empirical_ablations_unrerun",
    "paper_metrics_reproduced": false,
    "table6_mean_accuracy": {
      "64x1": 72.40875,
      "32x2": 77.55125000000001,
      "16x4": 79.62625,
      "6x10": 80.38125,
      "4x16": 80.04124999999999,
      "2x32": 71.86749999999999,
      "1x64": 67.1725
    },
    "table6_best_mean": "6x10",
    "table6_column_winners": [
      "6x10",
      "6x10",
      "6x10",
      "6x10",
      "6x10",
      "4x16",
      "6x10",
      "2x32"
    ],
    "table6_6x10_product": 60,
    "fixed_positional_channels_stated_in_paper": 384,
    "dimensions_implied_by_6x10_for_2d_simplex": 360,
    "official_released_scale_head_layout": "11 scales x 6 heads at width 396",
    "table7_mean_miou": {
      "2": 78.90285714285714,
      "100": 79.36428571428571,
      "1e4": 76.72571428571429,
      "1e6": 76.97857142857143
    },
    "table7_best_mean": "100",
    "table8_arithmetic": {
      "image": {
        "baseline_flops": 4.61,
        "ndrope_flops": 4.89,
        "baseline_params": 22.06,
        "ndrope_params": 23.36,
        "flops_increase_percent": 6.073752711496727,
        "params_increase_percent": 5.893019038984582
      },
      "video": {
        "baseline_flops": 196.05,
        "ndrope_flops": 196.05,
        "baseline_params": 121.0,
        "ndrope_params": 121.0,
        "flops_increase_percent": 0.0,
        "params_increase_percent": 0.0
      },
      "point_v1_vector": {
        "baseline_flops": 36.72,
        "ndrope_flops": 37.49,
        "baseline_params": 19.4,
        "ndrope_params": 19.4,
        "flops_increase_percent": 2.0969498910675455,
        "params_increase_percent": 0.0
      },
      "point_v2_vector": {
        "baseline_flops": 50.04,
        "ndrope_flops": 51.06,
        "baseline_params": 11.32,
        "ndrope_params": 11.33,
        "flops_increase_percent": 2.0383693045563644,
        "params_increase_percent": 0.08833922261484162
      }
    },
    "reason": "Table arithmetic supports 6x10 and theta=100 by mean, but checkpoints/configs for those ablations are absent; image FLOPs and parameters rise about 6%, driven by width 396 rather than learned frequency parameters."
  },
  "reproducibility_blockers": {
    "checkpoints": [],
    "semantic_kitti_files": [],
    "image_data_placeholder": true,
    "training_hardware_paper": "4 x NVIDIA A100 40GB",
    "paid_remote_compute_authorized": false
  },
  "runtime_seconds": 0.0510142499988433
}

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5d2c26bfd173", "created_at": "2026-07-19T17:16:51+00:00", "title": "Assessment"}
-->
**Anchored claim.** nD-RoPE reportedly reaches 75.85% Kinetics-400 top-1, 85.97% ModelNet40 mIoU, and 71.91% SemanticKITTI mIoU (Tables 2–4).

**Assessment: FALSIFIED AS WRITTEN.** The released entrypoints for the 85.97-mIoU experiment load `shapenetcore_partanno_segmentation_benchmark_v0_normal`, instantiate `PartNormalDataset`, use 16 ShapeNet object categories and 50 part labels, and compute ShapeNetPart instance-average part IoU. Their executable entrypoints never call `ModelNetDataLoader`; the separate ModelNet40 path is 40-way classification. The 154-file release also contains no SemanticKITTI implementation.

Because the anchored claim is conjunctive and assigns 85.97 mIoU to the wrong benchmark, one decisive false subclaim falsifies the claim as written. Kinetics and SemanticKITTI numbers remain untested rather than being declared false. The captured source audit hashes every decisive file and the independent verifier checks this conclusion fail-closed.
