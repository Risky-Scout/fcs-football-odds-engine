# Margin Target Contract

This document defines the canonical targets for the halftime / second-half / full-game margin PMF engine.

## Segment definitions

### First half

Use actual first-half score at halftime.

- `home_points_1h`
- `away_points_1h`

### Full game

Use final score including overtime only if the modeling choice explicitly includes OT.
Otherwise store both:

- regulation final score
- official final score

### Second half

Derived as:

- `home_points_2h = home_points_final - home_points_1h`
- `away_points_2h = away_points_final - away_points_1h`

## Margin targets

- `margin_1h = home_points_1h - away_points_1h`
- `margin_2h = home_points_2h - away_points_2h`
- `margin_game = home_points_final - away_points_final`

## Identity checks

The following must always hold:

- `home_points_game = home_points_1h + home_points_2h`
- `away_points_game = away_points_1h + away_points_2h`
- `margin_game = margin_1h + margin_2h`

## Market orientation

All spreads must be stored from the home-team perspective.

Examples:

- home `-3.5` means home favored by 3.5
- home `+7.0` means home underdog by 7.0

Then:

- home cover if `margin > -spread`
- away cover if `margin < -spread`

For non-push half-point lines, away cover is the complement of home cover.

## Required market fields

### Full game
- `market_spread_game`
- `market_total_game`

### First half
- `market_spread_1h`
- `market_total_1h`

### Second half
- `market_spread_2h`
- `market_total_2h`

## Recommended stored outcomes

- `actual_home_cover_game`
- `actual_home_cover_1h`
- `actual_home_cover_2h`
- `actual_over_game`
- `actual_over_1h`
- `actual_over_2h`

## QA requirements

A row is valid only if:

- all segment scores are nonnegative integers
- score identities reconcile
- margin identities reconcile
- spread orientation is consistent
- push behavior is explicit for integer lines
