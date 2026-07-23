"""Fail-closed checks for the additive Hugging Face release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from PIL import Image


MUTABLE_PROTECTED_PATHS = {"logbook.json"}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt"}
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


def flatten_pages(node: dict[str, object]) -> list[dict[str, object]]:
    pages = [node]
    for child in node.get("children", []):
        pages.extend(flatten_pages(child))
    return pages


def verify(root: Path) -> dict[str, object]:
    baseline_manifest = parse_manifest(
        root
        / ".openresearch/artifacts/baseline/"
        "judged-space-dcbbd49ee4487f62443820c7dfe2be11ae10af51.sha256"
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

    allowlist_path = root / "release/hf-space-upload-allowlist.txt"
    allowlist = [
        line.strip()
        for line in allowlist_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    allowlist_files_exist = all((root / path).is_file() for path in allowlist)
    text_only_allowlist = all(Path(path).suffix in TEXT_SUFFIXES for path in allowlist)
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
        "protected_revision": "dcbbd49ee4487f62443820c7dfe2be11ae10af51",
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
