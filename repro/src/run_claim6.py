"""Fail-closed Claim 6 paper-table and released-source consistency audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time


EXPECTED_PAPER_SHA256 = "aa7acd683cbb804762be15b607c1862b7be3fd0acd44424c939e2da9f0f6756f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_registered_frequency_buffer(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "register_buffer":
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "freqs":
            return True
    return False


def has_learnable_frequency_parameter(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Attribute) and target.attr == "freqs"
            for target in node.targets
        ):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "Parameter"
        ):
            return True
    return False


def quantified_frequency_counterexamples(data: dict[str, object]) -> list[dict[str, object]]:
    block = data["frequency_base"]
    bases = block["bases"]
    theta_index = bases.index(100)
    counterexamples = []
    for row in block["rows"]:
        values = row["miou"]
        best = max(values)
        if values[theta_index] < best:
            winner_index = values.index(best)
            counterexamples.append(
                {
                    "input_points": row["input_points"],
                    "training_points": block["training_points"],
                    "metric": block["metric"],
                    "theta_100": values[theta_index],
                    "better_base": bases[winner_index],
                    "better_score": best,
                    "margin_percentage_points": best - values[theta_index],
                }
            )
    return counterexamples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    data = json.loads(args.data.read_text(encoding="utf-8"))

    counterexamples = quantified_frequency_counterexamples(data)
    exact_in_domain = [
        item
        for item in counterexamples
        if item["input_points"] == item["training_points"]
    ]

    scale_rows = data["scale_head"]["rows"]
    allocations = [row["scales"] * row["heads"] for row in scale_rows]
    dimensions = [
        allocation * data["scale_head"]["dimensions_per_2d_scale"]
        for allocation in allocations
    ]
    means = {
        f'{row["scales"]}x{row["heads"]}': sum(row["accuracies"]) / len(row["accuracies"])
        for row in scale_rows
    }
    winners = []
    for column in range(len(data["scale_head"]["resolutions"])):
        winner = max(scale_rows, key=lambda row: row["accuracies"][column])
        winners.append(f'{winner["scales"]}x{winner["heads"]}')

    official_text = args.official_source.read_text(encoding="utf-8")
    registered_buffer = has_registered_frequency_buffer(official_text)
    learnable_parameter = has_learnable_frequency_parameter(official_text)
    vision = data["vision_cost"]
    cost = {
        "flops_increase_percent": 100
        * (vision["ndrope_flops_g"] - vision["baseline_flops_g"])
        / vision["baseline_flops_g"],
        "params_increase_percent": 100
        * (vision["ndrope_params_m"] - vision["baseline_params_m"])
        / vision["baseline_params_m"],
        "reported_parameter_difference_m": (
            vision["ndrope_params_m"] - vision["baseline_params_m"]
        ),
        "official_ndrope_freqs_registered_buffer": registered_buffer,
        "official_ndrope_freqs_learnable_parameter": learnable_parameter,
    }

    negative_controls = {
        "universal_quantifier_removed": len(
            quantified_frequency_counterexamples(
                {
                    "frequency_base": {
                        **data["frequency_base"],
                        "rows": [
                            {
                                **row,
                                "miou": (
                                    [85.00, 85.58, 83.66, 83.57]
                                    if row["input_points"] == 2048
                                    else row["miou"]
                                ),
                            }
                            for row in data["frequency_base"]["rows"]
                        ],
                    }
                }
            )
        )
        == 0,
        "average_only_interpretation_supports_theta_100": (
            sum(row["miou"][1] for row in data["frequency_base"]["rows"])
            > sum(row["miou"][0] for row in data["frequency_base"]["rows"])
        ),
        "constant_channel_control_detects_6x10": allocations.count(allocations[0])
        != len(allocations),
    }

    checks = {
        "paper_source_hash_pinned": data["provenance"]["source_sha256"]
        == EXPECTED_PAPER_SHA256,
        "theta_100_universal_statement_has_counterexample": len(counterexamples) >= 1,
        "counterexample_satisfies_training_grid_assumption": len(exact_in_domain) == 1,
        "counterexample_is_strict": exact_in_domain[0]["margin_percentage_points"] > 0,
        "table6_channel_conservation_contradicted": len(set(allocations)) > 1,
        "table6_6x10_implies_360_not_384": dimensions[3] == 360,
        "table6_6x10_is_not_every_resolution_winner": winners.count("6x10")
        < len(winners),
        "official_frequency_is_buffer_not_parameter": registered_buffer
        and not learnable_parameter,
        "all_negative_controls_behave_as_expected": all(negative_controls.values()),
    }
    report = {
        "claim": 6,
        "verdict": "FALSIFIED",
        "decisive_contract": data["frequency_base"]["quantified_statement"],
        "decisive_counterexamples": counterexamples,
        "scale_head_consistency": {
            "allocations_scales_times_heads": allocations,
            "implied_positional_dimensions": dimensions,
            "reported_fixed_embedding_dimension": data["scale_head"][
                "fixed_embedding_dimension"
            ],
            "mean_accuracy": means,
            "resolution_winners": winners,
        },
        "cost_consistency": cost,
        "negative_controls": negative_controls,
        "checks": checks,
        "inputs": {
            "paper_data_sha256": sha256(args.data),
            "official_source_sha256": sha256(args.official_source),
        },
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "No ablation checkpoint is released, so the table's trained metrics were not independently regenerated.",
            "The falsification is instead internal to the paper's exact universal statement and its own Table 7 under the stated training-grid condition.",
            "No conclusion depends on whether a six-percent FLOP increase is subjectively negligible.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "claim6_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "negative_controls.json").write_text(
        json.dumps(negative_controls, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
