from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "app.py",
    "MouseScope.spec",
    "BUILD_GUIDE.md",
    "README.md",
    "docs/system_design.md",
    "services/analysis_runtime.py",
    "services/video_metadata.py",
    "services/video_timebase.py",
    "core/tst_lis.py",
    "core/tst_legacy.py",
)

REQUIRED_GITIGNORE_PATTERNS = (
    "data/",
    "build/",
    "dist/",
    "__pycache__/",
    "*.sqlite",
    "*.db",
    ".pytest_cache/",
)

REQUIRED_SPEC_TOKENS = (
    "core",
    "templates",
    "static",
    "tst_lis",
    "tst_legacy",
    "services.video_metadata",
    "services.video_timebase",
)

STALE_RUNTIME_REFERENCES = (
    "core/tst_v2.py",
    "core\\tst_v2.py",
    "tst_v2_online_fit",
    "online_offline_diagnosis_cn.md",
)

SCAN_SUFFIXES = (".py", ".html", ".js", ".css", ".spec", ".md")
SCAN_ROOTS = (
    "app.py",
    "MouseScope.spec",
    "BUILD_GUIDE.md",
    "README.md",
    "core",
    "services",
    "routes",
    "templates",
    "static",
    "tests",
    "tools",
)

ALLOWED_STALE_REFERENCE_FILES = {
    Path("docs/system_design.md"),
    Path("tools/project_health_check.py"),
    Path("tests/test_project_health_check.py"),
}


@dataclass(frozen=True)
class Finding:
    severity: str
    message: str


def _run_git(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def _relative(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return path


def _iter_scan_files() -> Iterable[Path]:
    for item in SCAN_ROOTS:
        path = ROOT / item
        if path.is_file():
            if path.suffix.lower() in SCAN_SUFFIXES:
                yield path
            continue
        if not path.is_dir():
            continue
        for child in path.rglob("*"):
            if child.is_file() and child.suffix.lower() in SCAN_SUFFIXES:
                yield child


def check_required_files() -> list[Finding]:
    findings: list[Finding] = []
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            findings.append(Finding("ERROR", f"Required project file is missing: {rel}"))
    return findings


def check_gitignore() -> list[Finding]:
    path = ROOT / ".gitignore"
    if not path.is_file():
        return [Finding("ERROR", ".gitignore is missing")]
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = {line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")}
    findings: list[Finding] = []
    for pattern in REQUIRED_GITIGNORE_PATTERNS:
        if pattern not in lines:
            findings.append(Finding("ERROR", f".gitignore is missing safety pattern: {pattern}"))
    return findings


def check_tracked_user_data() -> list[Finding]:
    code, output = _run_git(["ls-files", "data"])
    if code != 0:
        return [Finding("WARN", f"Could not inspect tracked data files: {output}")]
    if not output:
        return []
    files = [line for line in output.splitlines() if line.strip()]
    preview = ", ".join(files[:5])
    suffix = "" if len(files) <= 5 else f", ... (+{len(files) - 5} more)"
    return [Finding("ERROR", f"User data appears to be tracked by git: {preview}{suffix}")]


def check_stale_runtime_references() -> list[Finding]:
    findings: list[Finding] = []
    for path in _iter_scan_files():
        rel = _relative(path)
        if rel in ALLOWED_STALE_REFERENCE_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            findings.append(Finding("WARN", f"Could not read {rel}: {exc}"))
            continue
        normalized = text.replace("\\", "/")
        reported_needles: set[str] = set()
        for token in STALE_RUNTIME_REFERENCES:
            needle = token.replace("\\", "/")
            if needle in reported_needles:
                continue
            if needle in normalized:
                reported_needles.add(needle)
                findings.append(Finding("WARN", f"Stale runtime reference '{token}' found in {rel}"))
    return findings


def check_spec_tokens() -> list[Finding]:
    path = ROOT / "MouseScope.spec"
    if not path.is_file():
        return [Finding("ERROR", "MouseScope.spec is missing")]
    text = path.read_text(encoding="utf-8", errors="replace")
    findings: list[Finding] = []
    for token in REQUIRED_SPEC_TOKENS:
        if token not in text:
            findings.append(Finding("ERROR", f"MouseScope.spec may miss required packaging token: {token}"))
    return findings


def check_dirty_worktree() -> list[Finding]:
    code, output = _run_git(["status", "--porcelain"])
    if code != 0:
        return [Finding("WARN", f"Could not inspect git status: {output}")]
    if not output:
        return []
    changed = len([line for line in output.splitlines() if line.strip()])
    return [Finding("WARN", f"Working tree has {changed} changed/untracked paths; freeze a release commit before packaging.")]


def check_large_files(threshold: int = 1200) -> list[Finding]:
    findings: list[Finding] = []
    for root_name in ("app.py", "core", "services", "routes"):
        root = ROOT / root_name
        files = [root] if root.is_file() else list(root.rglob("*.py")) if root.is_dir() else []
        for path in files:
            try:
                line_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
            except OSError:
                continue
            if line_count >= threshold:
                findings.append(Finding("WARN", f"Large file ({line_count} lines): {_relative(path)}"))
    return findings


def collect_findings() -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(check_required_files())
    findings.extend(check_gitignore())
    findings.extend(check_tracked_user_data())
    findings.extend(check_spec_tokens())
    findings.extend(check_stale_runtime_references())
    findings.extend(check_dirty_worktree())
    findings.extend(check_large_files())
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MouseScope release-oriented project health checks.")
    parser.add_argument("--strict-warnings", action="store_true", help="Treat warnings as failures.")
    args = parser.parse_args(argv)

    findings = collect_findings()
    errors = [item for item in findings if item.severity == "ERROR"]
    warnings = [item for item in findings if item.severity == "WARN"]

    if not findings:
        print("MouseScope project health check passed.")
        return 0

    print("MouseScope project health check findings:")
    for item in findings:
        print(f"[{item.severity}] {item.message}")

    if errors or (args.strict_warnings and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
