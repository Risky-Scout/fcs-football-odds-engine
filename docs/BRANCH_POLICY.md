# Branch Policy

## Branch roles

### main
Public best stable version only.

### feature/*
Used for new implementation work, experiments, and refactors.

### optional dev
Can be used as an integration branch if multiple feature branches are active at once.

## Rules

- never commit generated outputs or downloaded artifacts
- never push half-working experimental code directly to `main`
- only merge to `main` when:
  - tests or build scripts run
  - docs are updated
  - model status is updated if public state changed
  - the new version is actually the best public version

## Release habit

Tag public stable milestones, for example:
- `v0.1-reference-benchmark`
- `v0.2-margin-target-table`
- `v0.3-segment-pmf-baseline`
