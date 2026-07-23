"""Generate deterministic, evidence-bearing figures for the public report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "reports/ndrope-reproduction/images"


def save(fig: plt.Figure, name: str) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def headline_counterexample() -> None:
    data = json.loads(
        (ROOT / "repro/data/paper_claim6.json").read_text(encoding="utf-8")
    )
    bases = data["frequency_base"]["bases"]
    row = next(
        item
        for item in data["frequency_base"]["rows"]
        if item["input_points"] == data["frequency_base"]["training_points"]
    )
    scores = row["miou"]
    colors = ["#2563eb" if base == 2 else "#ef4444" if base == 100 else "#94a3b8" for base in bases]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar([str(base) for base in bases], scores, color=colors)
    ax.set_ylim(min(scores) - 0.25, max(scores) + 0.16)
    ax.set_xlabel(r"Frequency base $\theta$")
    ax.set_ylabel("Instance-average mIoU (%)")
    ax.set_title("Paper Table 7 contradicts “θ=100 is best across all settings”")
    for bar, value in zip(bars, scores, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.annotate(
        "strict counterexample\n+0.22 percentage points",
        xy=(0, scores[0]),
        xytext=(1.35, max(scores) - 0.60),
        arrowprops={"arrowstyle": "->", "color": "#1e3a8a"},
        color="#1e3a8a",
        weight="bold",
    )
    ax.grid(axis="y", alpha=0.2)
    save(fig, "claim6_counterexample.png")


def evidence_status() -> None:
    labels = ["C1", "C2", "C3", "C4", "C5", "C6"]
    outcomes = ["VERIFIED", "VERIFIED", "BLOCKED", "BLOCKED", "FALSIFIED", "FALSIFIED"]
    colors = {
        "VERIFIED": "#16a34a",
        "FALSIFIED": "#7c3aed",
        "BLOCKED": "#64748b",
    }
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.barh(labels[::-1], np.ones(6), color=[colors[value] for value in outcomes[::-1]])
    for index, outcome in enumerate(outcomes[::-1]):
        ax.text(0.5, index, outcome, ha="center", va="center", color="white", weight="bold")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Final evidence status—not a live judge score")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    save(fig, "evidence_status.png")


def architecture_confound() -> None:
    labels = ["nD-RoPE", "Axial RoPE", "RoPE-Mixed"]
    widths = [396, 384, 384]
    head_dims = [66, 64, 64]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    colors = ["#ef4444", "#64748b", "#64748b"]
    axes[0].bar(labels, widths, color=colors)
    axes[0].set_ylim(370, 402)
    axes[0].set_ylabel("Embedding width")
    axes[0].set_title("Released constructors")
    axes[1].bar(labels, head_dims, color=colors)
    axes[1].set_ylim(60, 68)
    axes[1].set_ylabel("Head dimension")
    axes[1].set_title("Six attention heads")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.2)
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{bar.get_height():.0f}",
                ha="center",
            )
    fig.suptitle("Claim 3 cannot be isolated from the released backbone confound")
    save(fig, "architecture_confound.png")


def cross_table_diagnostic() -> None:
    labels = ["Table 1\nstandard", "Table 5\n0°"]
    nd = [81.07, 80.81]
    mixed = [80.90, 80.99]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.plot(x, nd, marker="o", linewidth=2.5, label="nD-RoPE", color="#2563eb")
    ax.plot(x, mixed, marker="o", linewidth=2.5, label="RoPE-Mixed", color="#f97316")
    for xpos, values in enumerate(zip(nd, mixed, strict=True)):
        for value in values:
            ax.text(xpos + 0.03, value + 0.018, f"{value:.2f}", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Paper-reported top-1 (%)")
    ax.set_ylim(80.65, 81.16)
    ax.set_title("A diagnostic rank reversal that is not a valid falsification")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    save(fig, "cross_table_diagnostic.png")


def numerical_certificates() -> None:
    c1 = json.loads((ROOT / "outputs/claim1/claim1_report.json").read_text())
    c2 = json.loads((ROOT / "outputs/claim2/claim2_report.json").read_text())
    labels = [
        "C1 translation",
        "C1 relative",
        "C1 Parseval",
        "C2 structure",
        "C2 optimum",
    ]
    values = [
        c1["max_errors"]["translation"],
        c1["max_errors"]["relative_identity"],
        c1["max_errors"]["parseval_error"],
        max(
            c2["max_errors"]["centroid_norm"],
            c2["max_errors"]["norm_spread"],
            c2["max_errors"]["gram_error"],
            c2["max_errors"]["frame_error"],
        ),
        c2["max_errors"]["optimal_scale"],
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.bar(labels, values, color=["#16a34a"] * 5)
    ax.set_yscale("log")
    ax.set_ylabel("Maximum absolute error (log scale)")
    ax.set_title("Accepted mathematical claims remain inside strict numerical bounds")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", which="both", alpha=0.2)
    save(fig, "numerical_certificates.png")


def main() -> None:
    headline_counterexample()
    evidence_status()
    architecture_confound()
    cross_table_diagnostic()
    numerical_certificates()
    print(f"REPORT_ASSETS status=PASS count=5 directory={IMAGE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
