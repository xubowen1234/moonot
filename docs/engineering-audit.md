# Reproducible engineering inventory

Run `python scripts/audit_project.py --require-delivery` at the release commit.
The gate enumerates **tracked** production `.mbt` files, excluding blackbox and
whitebox tests, examples, benchmarks, executable entrypoints, vendor/generated
files and build outputs. Physical lines include blank lines as required by the
participant threshold. Nonblank and noncomment counts are reported separately
so the physical count cannot conceal padding. There is no generated dataset in
the production source inventory.

History authors and committers must both match the project-local identity.
`docs/competition/commit-audit.json` is a human-reviewed ledger: only reachable,
nonempty commits explicitly marked valid with purpose and verification count
toward the conservative total. The script does not pretend a nonempty diff
automatically proves engineering value. Metadata-only and delivery bookkeeping
are excluded even when they have real diffs.

Run `python -m unittest discover -s scripts -p test_audit_project.py` to verify
path exclusions and physical-line counting. Credential/history privacy has a
separate Skill gate and is not replaced by this inventory.
