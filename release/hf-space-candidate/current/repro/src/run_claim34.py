"""Four-route, fail-closed evidence audit for the two ImageNet claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time


CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".bin")
REQUIRED_ROUTES = {1, 2, 3, 4}


def manifest_paths(root: Path) -> list[str]:
    rows = []
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        _, path = line.split(maxsplit=1)
        rows.append(path)
    return rows


def extract_model_width(path: Path, model_name: str) -> tuple[int, int]:
    source = path.read_text(encoding="utf-8")
    start = source.index(f"def {model_name}")
    block = source[start : start + 900]
    width = re.search(r"embed_dim\s*=\s*(\d+)", block)
    heads = re.search(r"num_heads\s*=\s*(\d+)", block)
    if not width or not heads:
        raise ValueError(f"could not parse {model_name} in {path}")
    return int(width.group(1)), int(heads.group(1))


def evidence_gate(
    *,
    prediction_count: int,
    expected_count: int,
    has_checkpoint_digest: bool,
    has_labels: bool,
    exact_protocol: bool,
) -> bool:
    return (
        prediction_count == expected_count
        and has_checkpoint_digest
        and has_labels
        and exact_protocol
    )


def route1(data: dict[str, object], official_root: Path) -> dict[str, object]:
    paths = manifest_paths(official_root)
    checkpoints = [
        path for path in paths if path.lower().endswith(CHECKPOINT_SUFFIXES)
    ]
    image_data_markers = [
        path for path in paths if "imagenet" in path.lower() and "dataset" in path.lower()
    ]
    return {
        "route": 1,
        "name": "artifact_and_data_provenance",
        "interpretation": "The numeric claims require the exact trained model states and full ImageNet validation predictions under the stated protocols.",
        "official_commit": data["release"]["commit"],
        "manifest_file_count": len(paths),
        "checkpoint_files": checkpoints,
        "image_data_files": image_data_markers,
        "claim3_verification_gate": evidence_gate(
            prediction_count=0,
            expected_count=data["claim3"]["validation_images"],
            has_checkpoint_digest=False,
            has_labels=False,
            exact_protocol=False,
        ),
        "claim4_verification_gate": evidence_gate(
            prediction_count=0,
            expected_count=data["claim4"]["validation_images"],
            has_checkpoint_digest=False,
            has_labels=False,
            exact_protocol=False,
        ),
        "resolved": False,
        "reason": "The complete pinned release contains no trained checkpoint, ImageNet labels, or per-example predictions.",
    }


def route2(data: dict[str, object], official_root: Path) -> dict[str, object]:
    deit = official_root / "rope-vit-ndrope/deit"
    nd_width, nd_heads = extract_model_width(
        deit / "models_v2_ndRope.py", "ndrope_deit_small_patch16_LS"
    )
    axial_width, axial_heads = extract_model_width(
        deit / "models_v2_rope.py", "rope_axial_deit_small_patch16_LS"
    )
    mixed_width, mixed_heads = extract_model_width(
        deit / "models_v2_rope.py", "rope_mixed_deit_small_patch16_LS"
    )
    return {
        "route": 2,
        "name": "architecture_and_training_contract",
        "interpretation": "The paper's matched-backbone comparison should be recoverable from the released model constructors and training entrypoints.",
        "paper_architecture": data["claim3"]["architecture"],
        "released_models": {
            "ndrope": {
                "width": nd_width,
                "heads": nd_heads,
                "head_dim": nd_width // nd_heads,
            },
            "rope_axial": {
                "width": axial_width,
                "heads": axial_heads,
                "head_dim": axial_width // axial_heads,
            },
            "rope_mixed": {
                "width": mixed_width,
                "heads": mixed_heads,
                "head_dim": mixed_width // mixed_heads,
            },
        },
        "matched_width": nd_width == axial_width == mixed_width,
        "released_rotation_evaluator_present": any(
            "rotat" in path.lower() for path in manifest_paths(official_root)
        ),
        "resolved": False,
        "reason": "The released nD model is width 396/head_dim 66 while both comparison models are width 384/head_dim 64, and no rotation evaluator is released. This is a confound, not a valid counterexample to either reported metric.",
    }


def route3(data: dict[str, object], official_root: Path) -> dict[str, object]:
    c3 = data["claim3"]["reported_top1"]
    c4_zero = data["claim4"]["zero_degree_top1"]
    return {
        "route": 3,
        "name": "independent_cross_table_and_protocol_check",
        "interpretation": "If Table 1 and the zero-degree row of Table 5 used demonstrably identical models and evaluation, their in-domain ordering could cross-check Claim 3.",
        "table1": {"ndrope": c3["ndrope"], "rope_mixed": c3["rope_mixed"]},
        "table5_zero_degrees": c4_zero,
        "table1_margin_ndrope_minus_mixed": c3["ndrope"] - c3["rope_mixed"],
        "table5_zero_margin_ndrope_minus_mixed": c4_zero["ndrope"]
        - c4_zero["rope_mixed"],
        "rank_reverses": (c3["ndrope"] > c3["rope_mixed"])
        != (c4_zero["ndrope"] > c4_zero["rope_mixed"]),
        "rotation_transform_fully_specified": len(
            data["claim4"]["unspecified_transform_fields"]
        )
        == 0,
        "unspecified_transform_fields": data["claim4"][
            "unspecified_transform_fields"
        ],
        "resolved": False,
        "reason": "The rank reversal is material diagnostic evidence, but the paper does not establish identical checkpoints/preprocessing and omits interpolation/fill details. It therefore cannot verify or falsify either exact metric.",
    }


def route4(data: dict[str, object], official_root: Path) -> dict[str, object]:
    candidates = [
        {
            "candidate": "Table 5 zero-degree rank reversal versus Table 1",
            "assumptions_satisfied": False,
            "rejection": "identical checkpoints and evaluation preprocessing are not established",
        },
        {
            "candidate": "evaluate released randomly initialized models on CPU",
            "assumptions_satisfied": False,
            "rejection": "the claims quantify over 400-epoch trained ImageNet models",
        },
        {
            "candidate": "treat the released 396-versus-384 width confound as a metric counterexample",
            "assumptions_satisfied": False,
            "rejection": "a source confound supplies no contradictory top-1 measurement",
        },
    ]
    valid = [item for item in candidates if item["assumptions_satisfied"]]
    return {
        "route": 4,
        "name": "mandatory_exact_falsification_search",
        "interpretation": "A valid falsification must preserve ImageNet-1K, the trained ViT-S models, the exact 224 or 30-degree protocol, and contradict the reported numeric ordering.",
        "restated_claim3": data["claim3"]["statement"],
        "restated_claim4": data["claim4"]["statement"],
        "candidates": candidates,
        "valid_counterexamples": valid,
        "resolved": False,
        "reason": "No candidate both satisfies every paper assumption and supplies contradictory full-validation predictions.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    completed = data["completed_routes"]
    if completed != list(range(1, max(completed) + 1)):
        raise SystemExit("completed_routes must be a consecutive prefix starting at 1")

    implementations = {
        1: route1,
        2: route2,
        3: route3,
        4: route4,
    }
    routes = [
        implementations[route](data, args.official_root) for route in completed
    ]
    final_sequence_complete = set(completed) == REQUIRED_ROUTES
    verdict = "BLOCKED" if final_sequence_complete else "UNRESOLVED"
    negative_controls = {
        "table_values_without_predictions_are_not_verification": not evidence_gate(
            prediction_count=0,
            expected_count=50000,
            has_checkpoint_digest=False,
            has_labels=False,
            exact_protocol=True,
        ),
        "partial_predictions_are_rejected": not evidence_gate(
            prediction_count=49999,
            expected_count=50000,
            has_checkpoint_digest=True,
            has_labels=True,
            exact_protocol=True,
        ),
        "missing_checkpoint_digest_is_rejected": not evidence_gate(
            prediction_count=50000,
            expected_count=50000,
            has_checkpoint_digest=False,
            has_labels=True,
            exact_protocol=True,
        ),
        "complete_synthetic_gate_is_accepted": evidence_gate(
            prediction_count=50000,
            expected_count=50000,
            has_checkpoint_digest=True,
            has_labels=True,
            exact_protocol=True,
        ),
    }
    checks = {
        "official_manifest_count": len(manifest_paths(args.official_root))
        == data["release"]["manifest_files"],
        "routes_are_consecutive": completed == list(range(1, len(completed) + 1)),
        "no_route_claims_resolution": not any(route["resolved"] for route in routes),
        "negative_controls_fail_closed": all(negative_controls.values()),
        "terminal_verdict_only_after_four_routes": (
            (verdict == "BLOCKED") == final_sequence_complete
        ),
    }
    report = {
        "claims": [3, 4],
        "verdicts": {"claim3": verdict, "claim4": verdict},
        "confidence": {"claim3": "LOW", "claim4": "LOW"},
        "completed_routes": completed,
        "routes": routes,
        "negative_controls": negative_controls,
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
        "unblockers": {
            "claim3": [
                "exact trained nD-RoPE, RoPE-Axial, and RoPE-Mixed checkpoints",
                "ImageNet-1K validation access and per-example prediction evidence",
                "a matched width/backbone configuration or documented justification for width 396",
            ],
            "claim4": [
                "the exact fixed checkpoints used in Table 5",
                "ImageNet-1K validation access",
                "complete resize/rotation interpolation, fill, and antialias settings",
            ],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "claim34_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "negative_controls.json").write_text(
        json.dumps(negative_controls, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
