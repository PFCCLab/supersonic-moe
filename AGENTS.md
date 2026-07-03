# SonicMoE Agent Context

Use the repository indexes first, then open narrow targets.

## Canonical entrypoints

1. `INDEX.md` — root directory map with shallow overview and redundancy watchlist
2. `docs/HANDOFF.md` — authoritative project state, performance, architecture, and validation
3. `sonicmoe/INDEX.md` — source-package map
4. `tests/INDEX.md` — regression and operator test map
5. `tools/INDEX.md` — profiling / benchmark tooling map

## Mandatory index maintenance

- Before broad file search, read the nearest `INDEX.md`.
- Any file create / delete / rename / move must update the nearest affected `INDEX.md` summaries.
- Any edit that changes a file's responsibility must refresh that file summary in the same `INDEX.md`.
- Cross-directory moves must update both directory indexes and the nearest common ancestor index.
- After structural changes, regenerate indexes with `python tools/generate_directory_indexes.py` and review the generated summaries.

## Redundancy / historical notes

- `reports/fp8_upgrade/HANDOFF.md` is stale historical reference only; do not use it as the current handoff.
- `agent.md` exists only as a compatibility alias; edit `AGENTS.md` and keep `agent.md` thin.

## Confidentiality conventions

- **NEVER** commit absolute paths to the repository or local system in any source file, script, or configuration. Use relative paths or environment variables instead. (e.g., `/root/paddlejob/...` is a local path and should not be committed.)
- **NEVER** commit ERNIE-related business descriptions, production labels, private workload names, or private deployment context. Search keywords such as `ERNIE`, `ernie`, `ernie-core`, and `erniebot` as review signals, but this is not a blanket keyword ban: existing technical identifiers and compatibility-layer descriptions may keep saying ERNIE when that is the accurate technical contract. Use neutral terms such as reference shape or production-like workload for private workload descriptions.
