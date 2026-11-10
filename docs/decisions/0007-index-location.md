# 0007. The index lives in `.kibble/` inside the indexed repo

- Status: accepted
- Date: 2026-10-31

## Context
The index could live in a per-user cache keyed by repo path, or beside the code.

## Decision
A gitignored `.kibble/` directory in each indexed repo holds the SQLite file, lock
file and non-secret config. `kibble init` adds the `.gitignore` entry. Secrets are
never written there.

## Consequences
Easy to find, inspect and delete; moves with the repo. Pollutes the working tree
slightly and is per-checkout. A cache-directory mode can be added later behind
config.
