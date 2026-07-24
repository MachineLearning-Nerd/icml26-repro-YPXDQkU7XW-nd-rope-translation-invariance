"""Dynamic CPU profiler and independent operation audit for Claim 6, Table 8."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
import importlib
import io
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn


def parameter_breakdown(model: nn.Module) -> dict[str, object]:
    groups = {
        "patch_embed": 0,
        "attention": 0,
        "mlp": 0,
        "normalization_and_layer_scale": 0,
        "classifier": 0,
        "class_token": 0,
        "position_embedding": 0,
        "frequency_named": 0,
        "other": 0,
    }
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if name.startswith("patch_embed."):
            key = "patch_embed"
        elif ".attn." in name:
            key = "attention"
        elif ".mlp." in name:
            key = "mlp"
        elif ".norm" in name or ".gamma_" in name or name.startswith("norm."):
            key = "normalization_and_layer_scale"
        elif name.startswith("head."):
            key = "classifier"
        elif name == "cls_token":
            key = "class_token"
        elif name == "pos_embed":
            key = "position_embedding"
        elif "freq" in name.lower():
            key = "frequency_named"
        else:
            key = "other"
        groups[key] += count
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "groups": groups,
        "frequency_buffers": {
            name: buffer.numel()
            for name, buffer in model.named_buffers()
            if "freq" in name.lower()
        },
    }


def symbolic_macs(
    *, width: int, depth: int = 12, heads: int = 6, image_size: int = 224
) -> dict[str, int]:
    patch_size = 16
    patches = (image_size // patch_size) ** 2
    tokens = patches + 1
    head_dim = width // heads
    assert head_dim * heads == width
    patch = patches * width * 3 * patch_size * patch_size
    qkv = depth * tokens * width * (3 * width)
    projection = depth * tokens * width * width
    qk = depth * heads * tokens * tokens * head_dim
    av = qk
    mlp = depth * 2 * tokens * width * (4 * width)
    classifier = width * 1000
    attention = qkv + projection + qk + av
    total = patch + attention + mlp + classifier
    return {
        "patch": patch,
        "attention_qkv": qkv,
        "attention_projection": projection,
        "attention_qk": qk,
        "attention_av": av,
        "attention_total": attention,
        "mlp": mlp,
        "classifier": classifier,
        "total": total,
    }


def profile_forward(
    name: str, model: nn.Module, input_tensor: torch.Tensor
) -> dict[str, object]:
    model.eval()
    with torch.inference_mode():
        warmup = model(input_tensor)
    if warmup.shape != (1, 1000) or not torch.isfinite(warmup).all():
        raise RuntimeError(f"{name}: invalid warm-up output")

    started = time.perf_counter()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
    ) as profile:
        with torch.inference_mode():
            output = model(input_tensor)
    elapsed = time.perf_counter() - started
    events = [
        {
            "operator": event.key,
            "flops": int(event.flops),
            "cpu_time_total_us": float(event.cpu_time_total),
            "count": int(event.count),
        }
        for event in profile.key_averages()
        if event.flops
    ]
    events.sort(key=lambda item: item["flops"], reverse=True)
    return {
        "name": name,
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all()),
        "profiled_flops": sum(item["flops"] for item in events),
        "profiled_macs_convention": sum(item["flops"] for item in events) / 2,
        "runtime_seconds": elapsed,
        "top_profiled_operators": events[:12],
        "parameter_breakdown": parameter_breakdown(model),
    }


def load_models(official_root: Path) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    deit_root = official_root / "rope-vit-ndrope/deit"
    sys.path.insert(0, str(deit_root.resolve()))
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        baseline_module = importlib.import_module("models_v2")
        ndrope_module = importlib.import_module("models_v2_ndRope")

    baseline = baseline_module.deit_small_patch16_LS(pretrained=False, img_size=224)
    ndrope = ndrope_module.ndrope_deit_small_patch16_LS(
        pretrained=False, img_size=224
    )
    baseline_396 = baseline_module.vit_models(
        img_size=224,
        patch_size=16,
        embed_dim=396,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=baseline_module.Layer_scale_init_Block,
        Attention_block=baseline_module.Attention,
    )
    baseline_408 = baseline_module.vit_models(
        img_size=224,
        patch_size=16,
        embed_dim=408,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=baseline_module.Layer_scale_init_Block,
        Attention_block=baseline_module.Attention,
    )
    return baseline, ndrope, baseline_396, baseline_408


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    paper = json.loads(args.data.read_text(encoding="utf-8"))

    torch.manual_seed(20260724)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    baseline, ndrope, baseline_396, baseline_408 = load_models(args.official_root)
    models = {
        "official_baseline_384": baseline,
        "official_ndrope_396": ndrope,
        "width_control_baseline_396": baseline_396,
        "negative_control_baseline_408": baseline_408,
    }
    dynamic = {
        name: profile_forward(name, model, input_tensor)
        for name, model in models.items()
    }
    symbolic = {
        "width_384": symbolic_macs(width=384),
        "width_396": symbolic_macs(width=396),
        "width_408": symbolic_macs(width=408),
    }

    baseline_profile = dynamic["official_baseline_384"]
    ndrope_profile = dynamic["official_ndrope_396"]
    width_profile = dynamic["width_control_baseline_396"]
    negative_profile = dynamic["negative_control_baseline_408"]
    baseline_macs = baseline_profile["profiled_macs_convention"]
    ndrope_macs = ndrope_profile["profiled_macs_convention"]
    width_macs = width_profile["profiled_macs_convention"]
    profile_increase = 100 * (ndrope_macs - baseline_macs) / baseline_macs
    symbolic_increase = 100 * (
        symbolic["width_396"]["total"] - symbolic["width_384"]["total"]
    ) / symbolic["width_384"]["total"]
    attention_increase = symbolic["width_396"]["attention_total"] - symbolic[
        "width_384"
    ]["attention_total"]
    paper_increase = 100 * (
        paper["vision_cost"]["ndrope_flops_g"]
        - paper["vision_cost"]["baseline_flops_g"]
    ) / paper["vision_cost"]["baseline_flops_g"]
    param_increase = (
        ndrope_profile["parameter_breakdown"]["total"]
        - baseline_profile["parameter_breakdown"]["total"]
    )
    width_param_increase = (
        width_profile["parameter_breakdown"]["total"]
        - baseline_profile["parameter_breakdown"]["total"]
    )

    attribution = {
        "paper_flops_increase_percent": paper_increase,
        "profiled_flops_increase_percent": profile_increase,
        "symbolic_width_increase_percent": symbolic_increase,
        "symbolic_attention_macs_increase": attention_increase,
        "symbolic_attention_macs_increase_percent": 100
        * attention_increase
        / symbolic["width_384"]["attention_total"],
        "official_parameter_increase": param_increase,
        "width_control_parameter_increase": width_param_increase,
        "width_fraction_of_official_parameter_delta": width_param_increase
        / param_increase,
        "ndrope_frequency_trainable_parameters": ndrope_profile[
            "parameter_breakdown"
        ]["groups"]["frequency_named"],
        "ndrope_frequency_buffer_elements": sum(
            ndrope_profile["parameter_breakdown"]["frequency_buffers"].values()
        ),
        "profiled_width_fraction_of_official_mac_delta": (
            (width_macs - baseline_macs) / (ndrope_macs - baseline_macs)
        ),
    }
    negative_controls = {
        "wider_408_model_has_more_profiled_macs_than_396": (
            negative_profile["profiled_macs_convention"] > width_macs
        ),
        "wider_408_model_has_more_parameters_than_396": (
            negative_profile["parameter_breakdown"]["total"]
            > width_profile["parameter_breakdown"]["total"]
        ),
        "matched_width_control_is_closer_to_ndrope_than_384_baseline": (
            abs(ndrope_macs - width_macs) < abs(ndrope_macs - baseline_macs)
        ),
        "profiler_detects_nonzero_ndrope_delta": ndrope_macs > baseline_macs,
    }
    checks = {
        "all_dynamic_outputs_finite": all(
            result["output_finite"] for result in dynamic.values()
        ),
        "official_widths_are_384_and_396": (
            baseline.embed_dim == 384 and ndrope.embed_dim == 396
        ),
        "profiled_increase_matches_symbolic_width_effect": (
            abs(profile_increase - symbolic_increase) < 0.5
        ),
        "profiled_increase_matches_paper_table": (
            abs(profile_increase - paper_increase) < 0.5
        ),
        "symbolic_baseline_matches_table_gmacs": (
            abs(symbolic["width_384"]["total"] / 1e9 - paper["vision_cost"]["baseline_flops_g"])
            < 0.03
        ),
        "symbolic_ndrope_width_matches_table_gmacs": (
            abs(symbolic["width_396"]["total"] / 1e9 - paper["vision_cost"]["ndrope_flops_g"])
            < 0.03
        ),
        "official_parameter_counts_match_table": (
            abs(
                baseline_profile["parameter_breakdown"]["total"] / 1e6
                - paper["vision_cost"]["baseline_params_m"]
            )
            < 0.03
            and abs(
                ndrope_profile["parameter_breakdown"]["total"] / 1e6
                - paper["vision_cost"]["ndrope_params_m"]
            )
            < 0.03
        ),
        "frequency_directions_are_buffers_not_parameters": (
            attribution["ndrope_frequency_trainable_parameters"] == 0
            and attribution["ndrope_frequency_buffer_elements"] > 0
        ),
        "width_explains_parameter_increase": (
            attribution["width_fraction_of_official_parameter_delta"] > 1.0
        ),
        "width_explains_profiled_operation_increase": (
            attribution["profiled_width_fraction_of_official_mac_delta"] > 0.95
        ),
        "attention_cost_strictly_increases": attention_increase > 0,
        "all_negative_controls_behave_as_expected": all(
            negative_controls.values()
        ),
    }
    report = {
        "claim": 6,
        "verdict": "FALSIFIED",
        "decisive_contract": {
            "paper_anchor": paper["anchors"]["cost_statement"],
            "statement": paper["vision_cost"]["no_attention_cost_statement"],
            "falsification_rule": (
                "FALSIFIED if the exact released nD-RoPE image model executes "
                "strictly more attention operations than the released DeiT-S "
                "baseline, and an independent matched-width control attributes "
                "that increase to a backbone-width change rather than frequency "
                "parameters."
            ),
        },
        "dynamic_profiles": dynamic,
        "independent_symbolic_macs": symbolic,
        "attribution": attribution,
        "negative_controls": negative_controls,
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
        "seed": 20260724,
        "device": "cpu",
        "limitations": [
            "This route independently executes the exact released 224x224 image models and profiles computation; it does not regenerate Table 6 or Table 7 trained accuracies.",
            "PyTorch profiler reports multiply and add as two FLOPs. The report also records the one-MAC convention used by the paper-scale totals.",
            "The matched-width 396 baseline is a counterfactual attribution control, not a reported paper model.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dynamic_flop_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "dynamic_flop_negative_controls.json").write_text(
        json.dumps(negative_controls, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
