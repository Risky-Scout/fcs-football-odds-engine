# Possession Schema

## Purpose

This document defines the canonical possession-level training table for the FCS true-odds model.

**One row = one regulation possession.**

Overtime possessions must be stored separately and excluded from the regulation training set used by the main possession engine.

This table is the master source for:

- terminal possession outcome modeling
- PAT / 2-point modeling
- elapsed-time modeling
- next-field-position transition modeling
- regime labeling and late-game behavior analysis
- QA and audit workflows

---

## Core design principles

1. The possession table must be **deterministic and reproducible** from raw games, drives, and play-by-play.
2. The table must support a **coherent possession simulator** that ultimately generates the joint PMF:
   \[
   p(x,y)=P(\text{Team A}=x,\ \text{Team B}=y)
   \]
3. Ambiguous or unresolved possessions must be **flagged and excluded**, not forced into bad labels.
4. Terminal possession outcome and post-touchdown conversion are **modeled separately**.
5. Next starting field position must be represented both as an **exact yardline** and as a **5-yard modeling bin**.
6. Regulation and overtime must be handled as **separate state processes**.

---

## File outputs

Required outputs:

- `data/processed/possessions_regulation.parquet`
- `data/processed/possessions_overtime.parquet`

Supporting documentation:

- `docs/possession_schema.md`
- `docs/relabeling_rules.md`

---

## Canonical row definition

A canonical possession row must describe:

- who has the ball
- game state at possession start
- how the possession ends
- how many points are scored by offense, defense, and special teams
- whether a PAT or 2-point try follows
- how much clock the possession uses
- where the opponent starts next
- whether the possession belongs to a special regime such as clock-kill or desperation

---

## Field groups

### 1. Identity block

| Field | Type | Required | Description |
|---|---|---:|---|
| `game_id` | string | yes | Unique game identifier |
| `season` | int | yes | Season year |
| `week` | int | yes | Week number |
| `season_type` | string | yes | `regular` or `postseason` |
| `drive_id` | string | yes | Raw drive identifier |
| `possession_id` | string | yes | Canonical possession identifier |
| `offense_team` | string | yes | Team in possession at start |
| `defense_team` | string | yes | Opposing defense at start |
| `home_team` | string | yes | Home team |
| `away_team` | string | yes | Away team |
| `is_home_offense` | bool | yes | Whether offense is home team |

---

### 2. Start-state block

| Field | Type | Required | Description |
|---|---|---:|---|
| `period_start` | int | yes | Quarter at possession start |
| `clock_start_sec_in_period` | int | yes | Seconds remaining in current quarter |
| `clock_start_sec_in_half` | int | yes | Seconds remaining in current half |
| `clock_start_sec_in_game` | int | yes | Seconds remaining in regulation |
| `offense_score_start` | int | yes | Offense score at possession start |
| `defense_score_start` | int | yes | Defense score at possession start |
| `score_diff_start` | int | yes | `offense_score_start - defense_score_start` |
| `start_yardline_100` | int | yes | Yardline from offense perspective, 1 to 99 |
| `start_yards_to_goal` | int | yes | Usually same as `start_yardline_100` |
| `start_field_zone` | string | yes | `backed_up`, `own_side`, `midfield`, `fringe`, `red_zone` |
| `start_after_kickoff_flag` | bool | yes | Whether possession begins after kickoff |
| `start_after_turnover_flag` | bool | yes | Whether possession begins after turnover |
| `start_after_punt_flag` | bool | yes | Whether possession begins after punt |
| `start_after_fg_flag` | bool | yes | Whether possession begins after made or missed FG |
| `neutral_site_flag` | bool | yes | Whether the game is at a neutral site |

#### Start field zone definitions

| Zone | Rule |
|---|---|
| `backed_up` | start yardline 80-99 |
| `own_side` | start yardline 56-79 |
| `midfield` | start yardline 41-55 |
| `fringe` | start yardline 21-40 |
| `red_zone` | start yardline 1-20 |

---

### 3. Regime block

| Field | Type | Required | Description |
|---|---|---:|---|
| `late_half_flag` | bool | yes | Late-half situation indicator |
| `late_game_flag` | bool | yes | Late-regulation situation indicator |
| `clock_kill_flag` | bool | yes | Whether offense is primarily preserving time |
| `clock_kill_intensity` | int | yes | `0 = normal`, `1 = mild`, `2 = hard kill` |
| `kneel_sequence_flag` | bool | yes | Whether possession contains kneels |
| `desperation_flag` | bool | yes | Whether offense is in hurry-up / must-score regime |
| `is_overtime` | bool | yes | Whether possession occurs in overtime |

#### Regime notes

- Overtime possessions must be excluded from the main regulation training set.
- `clock_kill_flag` and `desperation_flag` are not mutually exclusive across the full dataset, but in practice should almost never both be `1` on the same possession.
- `kneel_sequence_flag` is a possession-content label, not just a start-state label.

---

### 4. Terminal target block

| Field | Type | Required | Description |
|---|---|---:|---|
| `terminal_result_raw` | string | yes | Raw drive result text from source |
| `terminal_result_model` | string | yes | Canonical 12-state label |
| `offensive_points_scored` | int | yes | Points scored by offense on this possession |
| `defensive_points_scored` | int | yes | Points scored by defense on this possession |
| `special_teams_points_scored` | int | yes | Points scored by special teams on this possession |
| `touchdown_flag` | bool | yes | Whether any TD occurred |
| `fg_attempt_flag` | bool | yes | Whether any field-goal attempt occurred |
| `turnover_flag` | bool | yes | Whether possession ended in a turnover |
| `turnover_on_downs_flag` | bool | yes | Whether possession ended on failed fourth down |
| `safety_flag` | bool | yes | Whether a safety occurred |
| `non_offensive_td_flag` | bool | yes | Whether a defensive or special-teams TD occurred |

#### Canonical 12-state taxonomy

`terminal_result_model` must be exactly one of:

1. `PUNT`
2. `FG_MADE`
3. `FG_MISS`
4. `OFF_TD`
5. `INT_LOST`
6. `FUMBLE_LOST`
7. `TURNOVER_ON_DOWNS`
8. `SAFETY_ALLOWED`
9. `DEF_TD_ALLOWED`
10. `ST_TD_ALLOWED`
11. `END_HALF`
12. `END_GAME`

No `OTHER` class is allowed in model training.

---

### 5. PAT / 2-point block

| Field | Type | Required | Description |
|---|---|---:|---|
| `post_td_try_type` | string | yes | `kick`, `two_point`, or `none` |
| `post_td_try_success` | int/null | yes | `1`, `0`, or `null` if no try |
| `post_td_points` | int | yes | `0`, `1`, or `2` |
| `post_td_sequence_complete` | bool | yes | Whether conversion sequence was resolved cleanly |

#### PAT / 2-point rules

- PAT / 2-point is never a terminal possession state.
- `OFF_TD`, `DEF_TD_ALLOWED`, and `ST_TD_ALLOWED` may all be followed by a conversion attempt.
- Non-touchdown possessions must have:
  - `post_td_try_type = none`
  - `post_td_try_success = null`
  - `post_td_points = 0`

---

### 6. Time block

| Field | Type | Required | Description |
|---|---|---:|---|
| `period_end` | int | yes | Quarter at possession end |
| `clock_end_sec_in_period` | int | yes | Seconds remaining in ending quarter |
| `clock_end_sec_in_game` | int | yes | Seconds remaining in regulation at possession end |
| `elapsed_seconds` | int | yes | Possession duration in seconds |

#### Time rules

- `elapsed_seconds` must be nonnegative.
- For regulation possessions:
  - `clock_end_sec_in_game <= clock_start_sec_in_game`
- End-of-half and end-of-game possessions are valid terminal states and may have no next possession.

---

### 7. Transition block

| Field | Type | Required | Description |
|---|---|---:|---|
| `next_possession_team` | string/null | yes | Team possessing next, or null if no next state |
| `next_start_yardline_exact` | int/null | yes | Exact next offensive start yardline |
| `next_start_field_bin_5` | string/null | yes | 5-yard field-position bin for next start |
| `next_state_exists` | bool | yes | Whether a next possession exists |
| `kickoff_to_next_flag` | bool | yes | Whether next possession begins after kickoff |
| `post_safety_restart_flag` | bool | yes | Whether next state follows safety restart mechanics |

#### 5-yard field-position bins

Allowed values for `next_start_field_bin_5`:

- `001_005`
- `006_010`
- `011_015`
- `016_020`
- `021_025`
- `026_030`
- `031_035`
- `036_040`
- `041_045`
- `046_050`
- `051_055`
- `056_060`
- `061_065`
- `066_070`
- `071_075`
- `076_080`
- `081_085`
- `086_090`
- `091_095`
- `096_099`

If `next_state_exists = false`, then:

- `next_possession_team = null`
- `next_start_yardline_exact = null`
- `next_start_field_bin_5 = null`

---

### 8. QA and audit block

| Field | Type | Required | Description |
|---|---|---:|---|
| `is_end_half_truncation` | bool | yes | Possession ends because half expires |
| `is_end_game_truncation` | bool | yes | Possession ends because regulation expires |
| `is_data_conflict` | bool | yes | Raw source conflict detected |
| `exclude_from_training` | bool | yes | Whether possession should be excluded from training |
| `source_confidence` | float | yes | Confidence score on `[0,1]` |
| `relabel_version` | string | yes | Version of relabeling rules used |
| `qa_notes` | string/null | yes | Audit notes if needed |

---

## Required invariants

The following invariants must hold for any possession included in training.

### Scoring reconciliation

\[
\Delta \text{score} =
\text{offensive_points_scored}
+
\text{defensive_points_scored}
+
\text{special_teams_points_scored}
\]

This total must reconcile with observed scoreboard change across the possession.

### Terminal-state consistency

- `FG_MADE` must imply `offensive_points_scored = 3`
- `FG_MISS` must imply `fg_attempt_flag = true`
- `OFF_TD` must imply `touchdown_flag = true`
- `INT_LOST` and `FUMBLE_LOST` must imply `turnover_flag = true`
- `TURNOVER_ON_DOWNS` must imply `turnover_on_downs_flag = true`
- `SAFETY_ALLOWED` must imply `defensive_points_scored = 2`
- `DEF_TD_ALLOWED` and `ST_TD_ALLOWED` must imply `non_offensive_td_flag = true`

### Conversion consistency

- Touchdown possessions must have valid PAT / 2-point fields
- Non-touchdown possessions must have `post_td_try_type = none`

### Transition consistency

- If `next_state_exists = true`, then `next_possession_team` must be non-null
- If `next_state_exists = false`, then all next-state fields must be null
- `next_start_field_bin_5` must match `next_start_yardline_exact`

### Overtime segregation

- Overtime possessions must not be included in the regulation training table

---

## Exclusion policy

Set `exclude_from_training = true` when any of the following is true:

- unresolved terminal state
- score delta does not reconcile
- impossible or broken clock transition
- next possession cannot be reconstructed reliably
- overtime possession appears in regulation table
- duplicate or broken drive identity
- PAT / 2-point sequence is unresolved in a way that affects scoring truth
- raw source conflict is severe enough to undermine trust

Excluded possessions must remain in the processed table for audit purposes.

Do not silently drop them.

---

## Recommended sort order

The possession table should be sorted by:

1. `season`
2. `week`
3. `game_id`
4. `clock_start_sec_in_game` descending within regulation flow
5. `possession_id`

This makes audit and replay easier.

---

## Modeling note

The possession table supports the following simulator sequence for a future game:

1. draw terminal possession outcome
2. draw PAT / 2-point result if applicable
3. draw elapsed time
4. draw next starting field position
5. update score, clock, and possession
6. repeat until regulation ends
7. simulate overtime separately if tied

That simulator ultimately estimates the joint PMF over final score pairs:
\[
p(x,y)=P(\text{Team A}=x,\ \text{Team B}=y)
\]

---

## Minimum acceptance criteria

A possession table build is considered acceptable only if:

- field definitions are complete
- terminal-state coverage is deterministic
- excluded rows are explicitly tracked
- QA checks pass at high rates
- transition fields are internally consistent
- touchdown conversion handling is separated cleanly
- overtime is separated from regulation

