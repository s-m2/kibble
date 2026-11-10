# 0006. npm for the JavaScript package

- Status: accepted
- Date: 2026-10-31

## Context
There is exactly one JS package. pnpm, Yarn and Bun optimize multi-package workspaces,
install speed and disk use, none of which matter here. Users launch the server with
`npx` and may use `bunx` or `pnpm dlx`.

## Decision
Plain npm for development, CI and publishing (with provenance via trusted
publishing). `package-lock.json` is committed; CI uses `npm ci`.

## Consequences
One familiar, documented workflow. Revisit if the repo grows more JS packages (npm
workspaces first). Before the first public release, add CI smoke tests that run the
published package under npx, bunx and pnpm dlx, so compatibility is verified instead
of assumed.
