# FCS Football Odds Engine

An FCS football modeling repo focused on estimating true probabilities and distributions better than the market, achieving strong calibration, and identifying positive-EV bets.

## Public repo policy

This repository is organized so that:

- `main` contains only the best stable public version
- heavy processing runs on GitHub Actions
- generated artifacts are not committed
- experimental work should happen on feature branches, not directly on `main`

## Current model layers

### Stable public layers
- historical FCS market dataset builder
- benchmark evaluation pipeline
- calibration pipeline
- reference model selection pipeline
- halftime / second-half / full-game margin target table
- halftime / full-game PMF specification scaffold

### Next implementation layer
- 1H / 2H / game segment PMF baseline
- linked score PMF and margin PMF evaluation
- later: latent team-state / possession-informed PMF engine

## Repo structure

- `docs/` — design specs, contracts, implementation order
- `configs/` — model configuration files
- `src/` — model and package source code
- `scripts/` — reproducible build/evaluation entrypoints
- `.github/workflows/` — heavy cloud pipelines

## Important rule

No result should be treated as production-quality unless it is:

- held-out validated
- calibration-checked
- benchmark-beating
- stable enough to justify public visibility on `main`
