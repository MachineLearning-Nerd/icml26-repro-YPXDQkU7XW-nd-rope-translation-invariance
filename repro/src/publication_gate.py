"""Fail-closed publication and evidence-integrity gate for nD-RoPE."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


EXPECTED_REPOSITORY = "icml26-nd-rope-translation-invariance"
EXPECTED_OFFICIAL_COMMIT = "f2cae70760806451f5e58be4b7e3dc4d0d856a1e"
EXPECTED_STATUSES = {
    "claim1": "verified",
    "claim2": "verified",
    "claim3": "BLOCKED",
    "claim4": "BLOCKED",
    "claim5": "falsified_as_written",
    "claim6": "FALSIFIED",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(root: Path, relative: str) -> dict[str, object]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def text_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.suffix.lower() not in {".png", ".pdf", ".svg", ".bin"}
    ]


def verify(root: Path) -> dict[str, object]:
    sources = load_json(root, "sources.json")
    claim1 = load_json(root, "outputs/claim1/claim1_report.json")
    claim2 = load_json(root, "outputs/claim2/claim2_report.json")
    claim34 = load_json(root, "outputs/claim34/claim34_report.json")
    source_audit = load_json(root, "outputs/source_audit/source_audit.json")
    claim6 = load_json(root, "outputs/claim6/claim6_report.json")
    verification = load_json(root, "outputs/verification.json")
    failure_controls = load_json(root, "outputs/verifier_failure_controls.json")
    release_gate = load_json(root, ".openresearch/artifacts/release/release_gate.json")

    source_checks = {
        artifact["path"]: sha256(root / artifact["path"]) == artifact["sha256"]
        for artifact in sources["artifacts"]
    }
    with tarfile.open(root / "sources/arxiv/source.tar.gz", mode="r:gz") as archive:
        source_archive_readable = any(
            member.name.endswith("ndrope.tex") for member in archive.getmembers()
        )

    text_hits: dict[str, list[str]] = {}
    patterns = {
        "absolute_path": re.compile(r"/" + r"(?:Users|home)/"),
        "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    }
    for path in text_files(root):
        if "vendor" in path.parts or path == root / "poster/GATE_REPORT.json":
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in patterns.items():
            if pattern.search(content):
                text_hits.setdefault(name, []).append(str(path.relative_to(root)))

    checks = {
        "repository_name": sources["paper"]["arxiv_id"] == "2606.12146"
        and EXPECTED_REPOSITORY in (root / "README.md").read_text(encoding="utf-8")
        and load_json(root, "sources.json")["official_implementation"]["commit"]
        == EXPECTED_OFFICIAL_COMMIT,
        "source_hashes": all(source_checks.values()),
        "source_archive_readable": source_archive_readable,
        "official_commit_pin": (root / "vendor/nD-RoPE/OFFICIAL_COMMIT").read_text(encoding="utf-8").strip()
        == EXPECTED_OFFICIAL_COMMIT,
        "claim1_verified": claim1["assessment"] == EXPECTED_STATUSES["claim1"]
        and all(claim1["checks"].values()),
        "claim2_verified": claim2["assessment"] == EXPECTED_STATUSES["claim2"]
        and all(claim2["checks"].values()),
        "claims34_blocked": claim34["verdicts"] == {
            "claim3": EXPECTED_STATUSES["claim3"],
            "claim4": EXPECTED_STATUSES["claim4"],
        }
        and claim34["completed_routes"] == [1, 2, 3, 4],
        "claim5_falsified": source_audit["claim5_cross_modal"]["assessment"]
        == EXPECTED_STATUSES["claim5"],
        "claim6_falsified": claim6["verdict"] == EXPECTED_STATUSES["claim6"],
        "independent_verifier": verification["all_checks_pass"] is True
        and failure_controls["all_mutations_rejected"] is True,
        "release_gate": release_gate["all_checks_pass"] is True,
        "no_top_level_trackio": not (root / ".trackio").exists(),
        "no_hygiene_hits": not text_hits,
    }
    report = {
        "publication_gate_passed": all(checks.values()),
        "repository": EXPECTED_REPOSITORY,
        "paper": "2606.12146",
        "checks": checks,
        "source_hashes": source_checks,
        "hygiene_hits": text_hits,
        "claim_statuses": EXPECTED_STATUSES,
    }
    destination = root / "outputs/publication_gate.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--skip-producers", action="store_true")
    result = verify(parser.parse_args().root.resolve())
    raise SystemExit(0 if result["publication_gate_passed"] else 1)


if __name__ == "__main__":
    main()
