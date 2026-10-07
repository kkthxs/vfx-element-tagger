#!/usr/bin/env python3
"""Export audited public source without private history, footage, models or local config."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".gitignore", ".python-version", ".vfx-tagger.example.json", "README.md", "LICENSE", "NOTICE",
    "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md", "MANIFEST.in", "pilot.py", "pyproject.toml",
}
PRIVATE = {"scripts/prepare_article_catalog.py", "scripts/stress_test_analysis.py", "constraints/macos-mlx-tested.txt"}
SENSITIVE = re.compile(
    r"/Users/|/Volumes/|[A-Za-z]:\\Users\\|"
    r"(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][^'\"\n]{12,}['\"]",
    re.IGNORECASE,
)


def public_files(root: Path) -> list[Path]:
    files = [root / name for name in sorted(ROOT_FILES) if (root / name).is_file()]
    for directory, suffixes in {
        "src": {".py", ".yaml", ".json"}, "scripts": {".py"}, "tests": {".py"},
        "docs": {".md"}, "constraints": {".txt"}, ".github": {".yml", ".md"},
    }.items():
        for path in sorted((root / directory).rglob("*")):
            relative = path.relative_to(root)
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in relative.parts:
                if str(relative) not in PRIVATE:
                    files.append(path)
    protocol = root / "evaluation/BENCHMARK-PROTOCOL.md"
    if protocol.is_file():
        files.append(protocol)
    return files


def scan_files(root: Path, files: list[Path]) -> list[str]:
    findings = []
    for path in files:
        # The scanner's own detection literals are not machine data or credentials.
        if path.name == "export_public_release.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if SENSITIVE.search(line):
                findings.append(f"{path.relative_to(root)}:{number}: private path or credential pattern")
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        parser.error("Public export must be outside the development repository")
    files = public_files(ROOT)
    findings = scan_files(ROOT, files)
    if findings:
        parser.error("Publication scan failed:\n" + "\n".join(findings))
    if args.check_only:
        print(f"PASS: {len(files)} allowlisted text files; no local paths or credential patterns")
        return
    if output.exists():
        parser.error("Output already exists; choose a fresh directory")
    output.mkdir(parents=True)
    entries = []
    for path in files:
        relative = path.relative_to(ROOT)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        entries.append({"path": str(relative), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    report = {
        "schema": 1, "files": entries, "checks": {"private_path_and_credential_patterns": "passed"},
        "excluded": ["development_git_history", "model_weights", "media", "catalogs", "local_config",
                     "private_evaluation_reports", "article_screenshots"],
        "limitations": "Pattern scan is not a guarantee against all secrets; inspect this exact tree before publishing.",
    }
    (output / "PUBLICATION-MANIFEST.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Exported {len(files)} source files; no Git history, media or model weights")


if __name__ == "__main__":
    main()
