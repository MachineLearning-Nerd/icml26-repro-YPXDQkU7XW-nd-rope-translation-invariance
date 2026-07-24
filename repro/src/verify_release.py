"""Fail-closed checks for the additive Hugging Face release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from PIL import Image


MUTABLE_PROTECTED_PATHS = {"README.md", "logbook.json"}
SECRET_PATTERNS = {
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "api_key_assignment": re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|secret)\b\s*[:=]\s*[\"'][^\"']{8,}"
    ),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        sha, relative = line.split(maxsplit=1)
        result[relative.removeprefix("./")] = sha
    return result


def is_utf8_text(path: Path) -> bool:
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    return True


def flatten_pages(node: dict[str, object]) -> list[dict[str, object]]:
    pages = [node]
    for child in node.get("children", []):
        pages.extend(flatten_pages(child))
    return pages


def verify(root: Path) -> dict[str, object]:
    baseline_manifest = parse_manifest(
        root
        / ".openresearch/artifacts/baseline/"
        "judged-space-f457f54c89151cc850279e904d28956e4c23508b.sha256"
    )
    candidate = root / "release/hf-space-candidate"
    candidate_paths = {
        str(path.relative_to(candidate)): path
        for path in candidate.rglob("*")
        if path.is_file()
    }
    protected_pages = {
        path for path in baseline_manifest if path.startswith("pages/")
    }
    path_subset = set(baseline_manifest).issubset(candidate_paths)
    byte_identical_protected_pages = all(
        digest(candidate_paths[path]) == baseline_manifest[path]
        for path in protected_pages
    )
    unexpected_protected_changes = sorted(
        path
        for path, expected in baseline_manifest.items()
        if path not in MUTABLE_PROTECTED_PATHS
        and path in candidate_paths
        and digest(candidate_paths[path]) != expected
    )

    logbook = json.loads((candidate / "logbook.json").read_text(encoding="utf-8"))
    pages = flatten_pages(logbook["root"])
    page_files = [page["file"] for page in pages]
    slugs = [page["slug"] for page in pages]
    tree_files_exist = all((candidate / str(path)).is_file() for path in page_files)
    unique_slugs = len(slugs) == len(set(slugs))
    current = candidate / "current"
    current_files = {
        str(path.relative_to(current)): path
        for path in current.rglob("*")
        if path.is_file()
    }
    current_pages = [
        current / "pages" / f"current-claim-{claim}.md"
        for claim in range(1, 7)
    ]
    current_root = current / "pages/current-verification.md"
    current_visibility = current / "pages/current-visibility.md"
    current_first = (
        logbook["root"]["file"] == "current/pages/current-verification.md"
        and [child["slug"] for child in logbook["root"]["children"][:6]]
        == [f"current-claim-{claim}" for claim in range(1, 7)]
    )
    historical_labeled = all(
        "Historical rejected baseline" in str(page["title"])
        for page in pages
        if str(page["file"]).startswith("pages/")
    )
    expected_visible_paths = [
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "repro/src/run_campaign.py",
        "repro/src/verify_results.py",
        "repro/src/run_verifier_failure_controls.py",
        "outputs/verification.json",
        "outputs/verifier_failure_controls.json",
        ".openresearch/artifacts/run_metadata.json",
        "source_revision.json",
    ]
    current_expected_files_exist = all(
        path in current_files for path in expected_visible_paths
    )
    current_text_only = all(is_utf8_text(path) for path in current_files.values())
    page_requirements = (
        "claim contract",
        "assumption",
        "Executable",
        "Raw",
        "Independent",
        "control",
        "Fixed command",
        "Limitation",
        "Run SHA",
    )
    current_claim_pages_complete = all(
        page.is_file()
        and all(
            requirement.casefold()
            in page.read_text(encoding="utf-8").casefold()
            for requirement in page_requirements
        )
        for page in current_pages
    )
    root_text = (
        current_root.read_text(encoding="utf-8")
        if current_root.is_file()
        else ""
    )
    current_root_complete = all(
        text in root_text
        for text in (
            "supersedes the historical rejected baseline",
            "uv run --frozen python repro/src/run_campaign.py",
            "outputs/verification.json",
            "outputs/verifier_failure_controls.json",
            ".openresearch/artifacts/run_metadata.json",
            "#/current-claim-1",
            "#/current-claim-6",
        )
    )
    visibility_text = (
        current_visibility.read_text(encoding="utf-8")
        if current_visibility.is_file()
        else ""
    )
    visibility_matrix_complete = (
        visibility_text.count("| Current Claim") == 0
        and all(f"| {claim} |" in visibility_text for claim in range(1, 7))
        and "Code visible" in visibility_text
        and "Reviewer verdict" in visibility_text
    )
    failure_controls_path = current / "outputs/verifier_failure_controls.json"
    failure_controls = (
        json.loads(failure_controls_path.read_text(encoding="utf-8"))
        if failure_controls_path.is_file()
        else {}
    )
    all_claim_mutations_rejected = (
        failure_controls.get("all_mutations_rejected") is True
        and len(failure_controls.get("cases", [])) == 6
        and all(
            case["verifier_exit_code"] != 0
            and not case["all_checks_pass"]
            and case["expected_check_failed"]
            for case in failure_controls.get("cases", [])
        )
    )

    allowlist_path = root / "release/hf-space-upload-allowlist.txt"
    allowlist = [
        line.strip()
        for line in allowlist_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    allowlist_files_exist = all((root / path).is_file() for path in allowlist)
    text_only_allowlist = all(is_utf8_text(root / path) for path in allowlist)
    changed_or_new = {
        f"release/hf-space-candidate/{path}"
        for path, file_path in candidate_paths.items()
        if path not in baseline_manifest or digest(file_path) != baseline_manifest[path]
    }
    allowlist_exact = set(allowlist) == changed_or_new

    upload_manifest_path = root / "release/hf-space-upload-manifest.sha256"
    upload_manifest = "\n".join(
        f"{digest(root / path)}  {path}" for path in allowlist
    ) + "\n"
    upload_manifest_path.write_text(upload_manifest, encoding="utf-8")

    secret_hits: dict[str, list[str]] = {}
    for path in allowlist:
        content = (root / path).read_text(encoding="utf-8")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                secret_hits.setdefault(name, []).append(path)

    images = sorted((root / "reports/ndrope-reproduction/images").glob("*.png"))
    valid_images = True
    for path in images:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            valid_images = False

    checks = {
        "old_file_set_is_subset": path_subset,
        "all_preexisting_pages_byte_identical": byte_identical_protected_pages,
        "only_allowlisted_old_routing_file_changed": unexpected_protected_changes == [],
        "logbook_space_id_exact": logbook["space_id"] == "DineshAI/YPXDQkU7XW",
        "logbook_tree_files_exist": tree_files_exist,
        "logbook_slugs_unique": unique_slugs,
        "current_verification_is_default_and_first": current_first,
        "historical_pages_explicitly_labeled": historical_labeled,
        "current_expected_source_and_evidence_exist": current_expected_files_exist,
        "current_bundle_is_utf8_text_only": current_text_only,
        "current_claim_pages_show_required_evidence": current_claim_pages_complete,
        "current_root_is_self_contained": current_root_complete,
        "visibility_matrix_has_six_complete_rows": visibility_matrix_complete,
        "all_six_claim_mutations_exit_nonzero": all_claim_mutations_rejected,
        "upload_allowlist_files_exist": allowlist_files_exist,
        "upload_allowlist_text_only": text_only_allowlist,
        "upload_allowlist_exactly_changed_or_new": allowlist_exact,
        "no_secret_patterns": not secret_hits,
        "five_report_images_present": len(images) == 5,
        "all_report_images_valid": valid_images,
    }
    report = {
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "protected_revision": "f457f54c89151cc850279e904d28956e4c23508b",
        "protected_file_count": len(baseline_manifest),
        "candidate_file_count": len(candidate_paths),
        "preexisting_page_count": len(protected_pages),
        "unexpected_protected_changes": unexpected_protected_changes,
        "allowlist": allowlist,
        "upload_manifest_sha256": digest(upload_manifest_path),
        "secret_hits": secret_hits,
    }
    destination = root / ".openresearch/artifacts/release/release_gate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    report = verify(parser.parse_args().root.resolve())
    raise SystemExit(0 if report["all_checks_pass"] else 1)


if __name__ == "__main__":
    main()
