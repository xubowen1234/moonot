# Delivery audit and reproducibility

Account/owner/namespace: `xubowen1234`; module: `xubowen1234/moonot@0.1.0`. Repository: https://github.com/xubowen1234/moonot. All reachable authors and committers use the project-selected name and GitHub noreply address. Author correction was explicitly authorized, content/time/order-preserving, and backed up outside the repository; it is not counted as engineering work.

## Local evidence (2026-10-14)

- Windows toolchain: Moon 0.1.20260920 (914d7da), core 0.10.14+7d59c7ec9.
- `moon info`, interface review and `moon fmt --check`.
- Strict `moon check --deny-warn`, build and test for wasm-gc, wasm and js: 190 tests on each target.
- Nine executable examples plus the CLI ran on each of these targets. The assertion-bearing alignment pipeline rejects solver/bridge/projection failures.
- Native strict checking passed. Local native build/test is **not** claimed: no C compiler is installed. The CI matrix runs those operations on Ubuntu.
- Independent POT harness: 60 cases, 56 actual POT comparisons and four analytic boundaries; 28 entropy/KL convention cross-checks. See `docs/reference-validation.md` for observed errors, pinned dependencies and limits.
- Production inventory: 3,362 physical MoonBit lines excluding tests, examples, generated interfaces and entrypoints. No generated data counts toward production scale. `scripts/audit_project.py` reports nonblank/noncomment counts too.
- `docs/competition/commit-audit.json`: 20 manually reviewed meaningful commits; package-discovery metadata and this delivery bookkeeping are excluded.

## Remote evidence boundary

The live [CI workflow](https://github.com/xubowen1234/moonot/actions/workflows/ci.yml), [releases](https://github.com/xubowen1234/moonot/releases) and exact public module metadata are the authoritative delivery evidence. Their existence or success is not inferred from local checks or this document. Publication requires a matching tested public SHA, green CI, privacy inspection of reachable history and real ZIP, and a no-credential new-consumer installation/check/run. The final local verification record contains the actual run/tag/package identifiers after those checks.

The Chinese application is kept outside this checkout and is never a repository/package/Release attachment. No registration form is submitted automatically. The recorded fixed participant deadline is 2026-10-30 24:00 (Asia/Shanghai).
