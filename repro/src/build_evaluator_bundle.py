"""Build the self-contained, text-only evaluator-visible bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


BINARY_VENDOR_PATHS = {
    "figures/overview.png",
    "figures/resolution_extrapolation_plots.png",
    "rope-vit-ndrope/.DS_Store",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        sha, relative = line.split(maxsplit=1)
        result[relative.removeprefix("./")] = sha
    return result


def copy_tree(
    source: Path,
    destination: Path,
    *,
    vendor: bool = False,
    excluded: set[str] | None = None,
    excluded_suffixes: tuple[str, ...] = (),
) -> None:
    def ignored(directory: str, names: list[str]) -> set[str]:
        directory_path = Path(directory)
        relative = directory_path.relative_to(source)
        ignored_names = {
            name
            for name in names
            if name == "__pycache__"
            or name.endswith((".pyc", ".pyo"))
            or name.endswith(excluded_suffixes)
            or str((relative / name).as_posix()) in (excluded or set())
            or (
                vendor
                and str((relative / name).as_posix()) in BINARY_VENDOR_PATHS
            )
        }
        return ignored_names

    shutil.copytree(source, destination, ignore=ignored)


def render_pages(root: Path, current: Path, revision: dict[str, object]) -> None:
    metadata = json.loads(
        (root / ".openresearch/artifacts/run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    verification = json.loads(
        (root / "outputs/verification.json").read_text(encoding="utf-8")
    )
    replacements = {
        "{{GIT_SHA}}": str(revision["git_sha"]),
        "{{TOTAL_RUNTIME_SECONDS}}": f"{metadata['total_runtime_seconds']:.6f}",
        "{{PLATFORM}}": str(metadata["platform"]),
        "{{CPU_COUNT}}": str(metadata["logical_cpu_count"]),
        "{{UV_LOCK_SHA256}}": str(metadata["uv_lock_sha256"]),
        "{{VERIFIER_CHECK_COUNT}}": str(len(verification["checks"])),
    }
    source = root / "release/evaluator-pages"
    destination = current / "pages"
    destination.mkdir()
    for template in sorted(source.glob("*.md")):
        text = template.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        unresolved = sorted(
            token
            for token in replacements
            if token in text
        )
        if unresolved:
            raise RuntimeError(
                f"unresolved page tokens in {template}: {unresolved}"
            )
        (destination / template.name).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    candidate = root / "release/hf-space-candidate"
    current = candidate / "current"
    if current.exists():
        shutil.rmtree(current)
    current.mkdir(parents=True)

    for name in ("pyproject.toml", "uv.lock", ".python-version", "EVAL.md"):
        shutil.copy2(root / name, current / name)
    copy_tree(
        root / "repro",
        current / "repro",
        excluded={
            "src/build_evaluator_bundle.py",
            "src/verify_release.py",
        },
    )
    copy_tree(root / "outputs", current / "outputs")
    copy_tree(root / "vendor/nD-RoPE", current / "vendor/nD-RoPE", vendor=True)
    copy_tree(
        root / ".openresearch/artifacts",
        current / ".openresearch/artifacts",
    )
    copy_tree(
        root / "reports/ndrope-reproduction",
        current / "reports/ndrope-reproduction",
        excluded_suffixes=(".png",),
    )
    copy_tree(root / "notebooks", current / "notebooks")
    (current / "reviews").mkdir()
    shutil.copy2(
        root / "release/evaluator-visible-review-initial.md",
        current / "reviews/initial_visibility_review.md",
    )
    revision = {
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "fixed_command": "uv run --frozen python repro/src/run_campaign.py",
        "evaluator_working_directory": "current",
        "official_source_commit": (
            root / "vendor/nD-RoPE/OFFICIAL_COMMIT"
        ).read_text(encoding="utf-8").strip(),
        "binary_vendor_reconstruction": (
            "Three non-text files are reconstructed from the pinned base64 "
            "payload before the fixed command audits the complete manifest."
        ),
    }
    (current / "source_revision.json").write_text(
        json.dumps(revision, indent=2) + "\n", encoding="utf-8"
    )
    render_pages(root, current, revision)

    protected = parse_manifest(
        root
        / ".openresearch/artifacts/baseline/"
        "judged-space-f457f54c89151cc850279e904d28956e4c23508b.sha256"
    )
    candidate_paths = {
        str(path.relative_to(candidate)): path
        for path in candidate.rglob("*")
        if path.is_file()
    }
    changed_or_new = sorted(
        f"release/hf-space-candidate/{relative}"
        for relative, path in candidate_paths.items()
        if relative not in protected or digest(path) != protected[relative]
    )
    allowlist = root / "release/hf-space-upload-allowlist.txt"
    allowlist.write_text("\n".join(changed_or_new) + "\n", encoding="utf-8")
    non_text_current = []
    for path in current.rglob("*"):
        if not path.is_file():
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            non_text_current.append(str(path.relative_to(current)))
    if non_text_current:
        raise RuntimeError(
            f"evaluator-visible bundle contains non-text files: {non_text_current}"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "current_file_count": sum(
                    1 for path in current.rglob("*") if path.is_file()
                ),
                "changed_or_new_file_count": len(changed_or_new),
                "binary_vendor_files_encoded_as_text": sorted(
                    BINARY_VENDOR_PATHS
                ),
                "fixed_command": revision["fixed_command"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
