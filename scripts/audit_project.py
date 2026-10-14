#!/usr/bin/env python3
"""Reproduce source inventory and check explicitly reviewed history, without network."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

EXCLUDED = {"examples", "benchmarks", "cmd", "vendor", "generated", "_build", ".git", ".mooncakes"}
EXPECTED_NAME = "xubowen1234"
EXPECTED_EMAIL = "279239855+xubowen1234@users.noreply.github.com"

def production_path(name):
    p = Path(name)
    return (p.suffix == ".mbt" and not p.name.endswith(("_test.mbt", "_wbtest.mbt"))
            and not any(part in EXCLUDED for part in p.parts))

def line_counts(text):
    lines = text.splitlines()
    return {"physical": len(lines), "nonblank": sum(bool(s.strip()) for s in lines),
            "nonblank_noncomment": sum(bool(s.strip()) and not s.lstrip().startswith("//") for s in lines)}

def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if result.returncode:
        raise ValueError("Git inspection failed")
    return result.stdout

def inspect(root, require_delivery=False):
    files = git(root, "ls-files", "-z").decode("utf-8").split("\0")
    inventory = []
    for name in sorted(filter(production_path, files)):
        inventory.append(dict(path=name, **line_counts((root / name).read_text(encoding="utf-8-sig"))))
    records = git(root, "log", "--format=%H%x00%an%x00%ae%x00%cn%x00%ce").decode("utf-8").splitlines()
    commits = {}
    errors = []
    for record in records:
        sha, author, email, committer, committer_email = record.split("\0")
        commits[sha] = record
        if (author, email, committer, committer_email) != (EXPECTED_NAME, EXPECTED_EMAIL, EXPECTED_NAME, EXPECTED_EMAIL):
            errors.append("Author/committer mismatch: " + sha)
    audited = []
    audit = root / "docs/competition/commit-audit.json"
    if audit.exists():
        entries = json.loads(audit.read_text(encoding="utf-8"))["commits"]
        seen = set()
        for entry in entries:
            sha = entry["sha"]
            if sha in seen or sha not in commits:
                errors.append("Duplicate or unreachable reviewed commit: " + sha)
                continue
            seen.add(sha)
            if entry.get("valid") is not True:
                continue
            if not entry.get("purpose") or not entry.get("verification"):
                errors.append("Reviewed commit lacks rationale/evidence: " + sha)
                continue
            diff = git(root, "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", sha).strip()
            if not diff:
                errors.append("Reviewed commit is empty: " + sha)
                continue
            audited.append(sha)
    physical = sum(item["physical"] for item in inventory)
    if require_delivery:
        if physical < 1500:
            errors.append("Production physical lines below 1500")
        if len(audited) < 20:
            errors.append("Explicitly reviewed meaningful commits below 20")
    return {"production_inventory": inventory, "production_physical_lines": physical,
            "raw_commits": len(commits), "reviewed_meaningful_commits": len(audited), "errors": errors}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--require-delivery", action="store_true")
    args = parser.parse_args()
    try:
        result = inspect(args.project.resolve(), args.require_delivery)
    except (OSError, ValueError, KeyError) as exc:
        print("Audit failed: " + type(exc).__name__)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return int(bool(result["errors"]))

if __name__ == "__main__":
    sys.exit(main())
