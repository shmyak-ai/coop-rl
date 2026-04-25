---
name: Python Editing Scope & Safety
description: "Use when writing or refactoring Python in this repository. Enforce strict scope: do not change lockfiles or package metadata unless explicitly requested, and follow concise Python quality rules."
applyTo: "**/*.py"
---

# Python Editing Scope & Safety

## Scope Boundaries (Hard Rules)

- Change only files required by the user request.
- Do not modify dependency lock or package metadata files unless the user explicitly asks.
- Treat these as protected by default:
  - `uv.lock`
  - `requirements*.txt` (unless dependency changes were explicitly requested)
  - `*.egg-info/**`
  - generated metadata/build artifacts with equivalent role
- If a Python code change seems to require edits to protected files, stop and ask first.

## Python Code Rules

- Keep changes minimal and directly traceable to the request.
- Prefer simple, composable functions; use classes only when state/lifecycle boundaries are clearer.
- Add or preserve type annotations for function parameters and return values.
- Keep side effects at boundaries (I/O, logging, env interaction); keep core logic explicit.
- Match existing project style and import ordering; avoid unrelated refactors.
- Remove only imports/variables made unused by your own changes.

## Before Finalizing

- Verify no protected-file edits were introduced unintentionally.
- If a protected-file update is needed, ask for approval and explain why.
