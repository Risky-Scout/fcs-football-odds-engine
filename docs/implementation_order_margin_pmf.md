# Implementation Order: Halftime / Full-Game Margin PMF Engine

## Goal

Move from benchmark screening to the actual production-class probability engine.

## Order of work

### Phase 1: target engineering

Build historical target tables containing:

- halftime team scores
- second-half team scores
- full-game team scores
- halftime margin
- second-half margin
- full-game margin

### Phase 2: segment score distributions

Build segment-level score PMFs for:

- first half
- second half

### Phase 3: linked PMF construction

Combine first-half and second-half score PMFs into:

- full-game score PMF
- halftime margin PMF
- second-half margin PMF
- full-game margin PMF

### Phase 4: market pricing

Derive fair prices for:

- game spread
- first-half spread
- second-half spread
- game total
- first-half total
- second-half total

### Phase 5: held-out validation

Evaluate against:

- calibrated Pi benchmark
- market assumptions
- EV backtest

## Non-negotiable checkpoint

The new PMF engine should not be promoted unless it beats the benchmark reference on held-out spread performance.
