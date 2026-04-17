# Halftime / Full-Game Margin PMF Specification

## Mission

Build an FCS football pregame model that estimates true probabilities and distributions better than the market, is well-calibrated, and identifies positive-EV bets.

The model must produce exemplary pregame probability estimates for:

- full-game spread
- first-half spread
- second-half spread
- full-game moneyline
- halftime moneyline
- game total
- first-half total
- second-half total

The centerpiece is the **pregame probability mass function over scoring and margin states**.

---

## Core modeling objective

The production engine must estimate, before kickoff:

1. a halftime score PMF
2. a full-game score PMF
3. a halftime margin PMF
4. a full-game margin PMF
5. a second-half margin PMF

with structural consistency:

- `M_game = M_1H + M_2H`
- `Score_game = Score_1H + Score_2H`
- team-level score distributions and margin distributions must reconcile

This means the model is not just a side picker. It is a **distribution engine**.

---

## Primary outputs

For each pregame matchup, the engine must output at minimum:

### Full-game outputs

- `P(Home points = x, Away points = y)` for regulation + overtime handling choice
- `P(M_game = m)` where `m = home_score - away_score`
- fair win probabilities
- fair spread cover probabilities for arbitrary market spreads
- fair total probabilities for arbitrary market totals
- fair team total probabilities

### Halftime outputs

- `P(Home 1H points = x, Away 1H points = y)`
- `P(M_1H = m)`
- fair first-half spread probabilities
- fair first-half total probabilities

### Second-half outputs

- `P(Home 2H points = x, Away 2H points = y)`
- `P(M_2H = m)`
- fair second-half spread probabilities
- fair second-half total probabilities

---

## Modeling identity

The model should be built around **linked score and margin distributions**, not just independent market classifiers.

Recommended production identity:

1. pregame latent team-state model
2. possession-based scoring engine
3. halftime state projection
4. linked first-half / second-half / full-game PMF construction
5. post-hoc calibration layer where justified by held-out results

---

## Distribution constraints

The PMF engine must satisfy all of the following:

- probabilities are nonnegative
- probabilities sum to 1
- halftime, second-half, and full-game score distributions reconcile
- margin PMFs implied by score PMFs match directly computed margin PMFs
- spread and moneyline prices derived from PMFs reconcile with each other
- calibration is measured on held-out data

---

## Margin definitions

Let:

- `M_game = Home_final - Away_final`
- `M_1H = Home_halftime - Away_halftime`
- `M_2H = (Home_final - Home_halftime) - (Away_final - Away_halftime)`

Then:

- `M_game = M_1H + M_2H`

This identity is mandatory and must hold exactly in data engineering and in model outputs.

---

## Recommended factorization

The production engine should target this factorization:

1. estimate halftime possession volume and scoring intensity
2. estimate second-half possession volume and scoring intensity
3. estimate team scoring distributions within each segment
4. combine segment score PMFs into:
   - halftime score PMF
   - second-half score PMF
   - full-game score PMF
5. derive margin PMFs from score PMFs
6. calibrate only when held-out evidence supports calibration

A practical first production target is:

- estimate `P(Home_1H=x, Away_1H=y)`
- estimate `P(Home_2H=x, Away_2H=y)`
- combine them by convolution into full-game score PMF
- derive `P(M_1H)`, `P(M_2H)`, and `P(M_game)`

---

## Candidate production architecture

### Layer A: latent pregame team state

Weekly latent dimensions may include:

- offense
- defense
- pace
- explosiveness
- finishing drives
- turnover skill
- special teams

These states evolve week to week with partial pooling.

### Layer B: segment-level scoring engine

Separate segment models for:

- first half
- second half

Each segment should condition on:

- team latent states
- opponent latent states
- venue / home field
- tempo mismatch
- expected possession count
- expected field-position advantage
- expected game state pressure

### Layer C: score PMF generator

For each segment:

- generate team score PMFs or joint score PMFs

Then combine:

- first-half PMF
- second-half PMF
- full-game PMF by convolution or simulation

### Layer D: derived market probabilities

From the PMF, derive:

- moneyline probabilities
- arbitrary spread cover probabilities
- arbitrary total probabilities
- team totals
- alternate lines

### Layer E: calibration and EV logic

Only after the raw PMF is validated:

- apply market-specific calibration where held-out evidence justifies it
- compare fair probabilities to market implied probabilities
- compute edge and expected value

---

## Why this is the correct next model

The benchmark layer showed useful structure, but it did not produce a robust market-beating spread engine.

To beat the market on:

- game spread
- first-half spread
- second-half spread

the model must estimate the **shape** of the margin distribution, not just a single expected spread number.

That is why the next step is a margin PMF engine.

---

## Required training targets

The modeling stack must be able to build these supervised targets from historical data:

### Score targets

- home halftime points
- away halftime points
- home second-half points
- away second-half points
- home full-game points
- away full-game points

### Margin targets

- halftime margin
- second-half margin
- full-game margin

### Binary market targets

- home full-game cover at market spread
- home first-half cover at market first-half spread
- home second-half cover at market second-half spread
- over full-game total
- over first-half total
- over second-half total

---

## Evaluation hierarchy

The model should be evaluated in this order:

1. score-distribution coherence
2. margin-distribution coherence
3. log loss / Brier / calibration for spread and total probabilities
4. held-out EV backtest against market assumptions
5. stability by season
6. comparison to benchmark references

---

## Acceptance criteria for production candidacy

The halftime/full-game margin PMF engine is production-candidate only if:

- it outperforms benchmark reference on held-out game spread metrics
- it outperforms benchmark reference on held-out total metrics
- it is well-calibrated on spread and total markets
- it produces stable season-by-season results
- its candidate EV signals survive out-of-sample backtesting
- its 1H / 2H / game distributions reconcile exactly

---

## Non-negotiable rule

No probability used for pricing or EV logic should be trusted merely because it looks sharp.

It must be:

- structurally coherent
- held-out validated
- calibration-checked
- benchmark-beating

That is the standard for the final FCS odds engine.
