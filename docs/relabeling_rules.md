# Relabeling Rules

## Purpose

This document defines the deterministic mapping from raw game, drive, and play-by-play data into the canonical possession labels used by the FCS true-odds model.

The relabeling engine exists to support the mission:

- estimate true probabilities and distributions better than the market
- produce a well-calibrated joint PMF over final score pairs
- identify positive-EV bets from fair odds derived from that PMF

Because the model is only as good as its possession labels, ambiguous possessions must be flagged and excluded rather than forced into incorrect classes.

---

## Core principle

Each regulation possession must be assigned exactly one canonical terminal state:

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

If a possession cannot be confidently mapped, set:

- `is_data_conflict = true`
- `exclude_from_training = true`

---

## Source precedence

When sources disagree, use this order of trust:

1. play-by-play sequence
2. drive-level result
3. scoreboard delta
4. fallback text fields

The final label must reflect the actual football event, not merely the raw drive description.

---

## Possession relabeling workflow

For each raw drive:

1. identify the offense and defense at possession start
2. collect all plays belonging to that drive
3. reconstruct the terminal football event
4. compute scoreboard change during the possession
5. separate touchdown conversion events from terminal state
6. infer the next offensive start state
7. run QA checks
8. keep or exclude the possession

---

## Deterministic raw-to-model mappings

### Direct mappings

| Raw outcome | Canonical label |
|---|---|
| Punt | `PUNT` |
| Field goal good | `FG_MADE` |
| Field goal missed / no good | `FG_MISS` |
| Offensive rushing touchdown | `OFF_TD` |
| Offensive passing touchdown | `OFF_TD` |
| Interception with no return TD | `INT_LOST` |
| Lost fumble with no return TD | `FUMBLE_LOST` |
| Turnover on downs | `TURNOVER_ON_DOWNS` |
| Safety charged to offense | `SAFETY_ALLOWED` |
| Half expires | `END_HALF` |
| Regulation expires | `END_GAME` |

### Override mappings from play detail

| Special case in play-by-play | Canonical label |
|---|---|
| Interception returned for touchdown | `DEF_TD_ALLOWED` |
| Fumble returned for touchdown by defense | `DEF_TD_ALLOWED` |
| Blocked punt returned for touchdown | `ST_TD_ALLOWED` |
| Blocked field goal returned for touchdown | `ST_TD_ALLOWED` |
| Punt return touchdown tied to the change of possession | `ST_TD_ALLOWED` |

---

## State-by-state relabeling rules

## 1. `PUNT`

Assign `PUNT` when:

- offense voluntarily ends the possession with a punt
- no touchdown is scored on the punt return
- no data conflict overrides the punt result

Required flags:

- `fg_attempt_flag = false`
- `turnover_flag = false`
- `touchdown_flag = false` unless the source is wrong, in which case exclude
- `non_offensive_td_flag = false`

Do not use `PUNT` if the punt is blocked and returned for a touchdown. That is `ST_TD_ALLOWED`.

---

## 2. `FG_MADE`

Assign `FG_MADE` when:

- a field goal attempt is successful
- the possession ends with 3 offensive points
- no touchdown is scored during the sequence

Required flags:

- `fg_attempt_flag = true`
- `offensive_points_scored = 3`

If source text says field goal good but score delta is not +3, exclude unless the play sequence resolves the contradiction cleanly.

---

## 3. `FG_MISS`

Assign `FG_MISS` when:

- a field goal attempt fails
- no touchdown is scored on the return
- possession changes because the kick is dead or returned without a touchdown

Required flags:

- `fg_attempt_flag = true`

Do not use `FG_MISS` if the blocked or missed kick is returned for a touchdown. That is `ST_TD_ALLOWED`.

---

## 4. `OFF_TD`

Assign `OFF_TD` when:

- the offense scores a touchdown on the possession
- the touchdown is not a defensive return touchdown
- the touchdown is not a special-teams touchdown attributed to the change of possession event

Important:

- `OFF_TD` means the offense scored 6 points
- PAT or 2-point attempt is modeled separately
- the terminal label does not include the conversion result

Required flags:

- `touchdown_flag = true`
- `non_offensive_td_flag = false`

---

## 5. `INT_LOST`

Assign `INT_LOST` when:

- the offense throws an interception
- the defense gains possession
- the play is not returned for a touchdown

Required flags:

- `turnover_flag = true`

Do not use this label for pick-sixes. Those are `DEF_TD_ALLOWED`.

---

## 6. `FUMBLE_LOST`

Assign `FUMBLE_LOST` when:

- the offense loses the ball by fumble
- the opponent gains possession
- the return does not score a touchdown

Required flags:

- `turnover_flag = true`

Do not use this label for fumble-return touchdowns. Those are `DEF_TD_ALLOWED`.

---

## 7. `TURNOVER_ON_DOWNS`

Assign `TURNOVER_ON_DOWNS` when:

- the offense fails to convert on fourth down
- the defense takes over
- the possession does not end by interception, fumble, punt, field goal, or expiration of time

Required flags:

- `turnover_on_downs_flag = true`

---

## 8. `SAFETY_ALLOWED`

Assign `SAFETY_ALLOWED` when:

- the offense concedes a safety
- the defense is awarded 2 points

Required flags:

- `safety_flag = true`
- `defensive_points_scored = 2`

The next possession after a safety follows special restart logic and must be flagged accordingly.

---

## 9. `DEF_TD_ALLOWED`

Assign `DEF_TD_ALLOWED` when:

- the offense allows a defensive touchdown
- the scoring event is caused by a defensive return after an interception or lost fumble

Examples:

- pick-six
- fumble return touchdown by defense

Required flags:

- `touchdown_flag = true`
- `non_offensive_td_flag = true`

Post-touchdown conversion must still be recorded separately.

---

## 10. `ST_TD_ALLOWED`

Assign `ST_TD_ALLOWED` when:

- the offense allows a special-teams touchdown during the possession sequence
- the touchdown occurs on a blocked kick return or return sequence tied to the change of possession

Examples:

- blocked punt returned for touchdown
- blocked field goal returned for touchdown
- punt return touchdown attributed to the kicking possession transition

Required flags:

- `touchdown_flag = true`
- `non_offensive_td_flag = true`

Post-touchdown conversion must still be recorded separately.

---

## 11. `END_HALF`

Assign `END_HALF` when:

- the possession ends because time expires in the half
- no other football event cleanly terminates the possession first

This is a real terminal state, not a missing-data condition.

Typical examples:

- offense is tackled in bounds as half expires
- incomplete pass with zeroes at end of half
- kneel or low-aggression sequence ending the half

---

## 12. `END_GAME`

Assign `END_GAME` when:

- the possession ends because regulation time expires
- no other terminal event takes priority

This is also a real terminal state.

Typical examples:

- final play ends in bounds and regulation expires
- offense kneels out the game
- desperation play ends with no further time remaining

Overtime possessions must not be labeled `END_GAME` in the regulation table.

---

## Touchdown conversion rules

Touchdown conversion is never part of the terminal state.

Populate the following separately:

- `post_td_try_type`
- `post_td_try_success`
- `post_td_points`

Allowed values:

- `post_td_try_type = kick`
- `post_td_try_type = two_point`
- `post_td_try_type = none`

Scoring rules:

- made kick = 1 point
- failed kick = 0 points
- successful 2-point try = 2 points
- failed 2-point try = 0 points

For non-touchdown possessions:

- `post_td_try_type = none`
- `post_td_try_success = null`
- `post_td_points = 0`

If the scoreboard moved but the play text for the conversion is missing, infer the conversion from the score delta when possible. If not possible, exclude the possession.

---

## Score reconciliation rules

For every possession, total scoring must reconcile with scoreboard change:

\[
\Delta \text{score} =
\text{offensive_points_scored}
+
\text{defensive_points_scored}
+
\text{special_teams_points_scored}
+
\text{post_td_points}
\]

If the relabeled possession does not reconcile with observed score movement, mark:

- `is_data_conflict = true`
- `exclude_from_training = true`

---

## Next-state reconstruction rules

Each possession must attempt to reconstruct the opponent's next offensive start.

Populate:

- `next_possession_team`
- `next_start_yardline_exact`
- `next_start_field_bin_5`
- `next_state_exists`

For `END_HALF` and `END_GAME`:

- `next_state_exists = false`
- `next_possession_team = null`
- `next_start_yardline_exact = null`
- `next_start_field_bin_5 = null`

Train the transition model on the 5-yard bin, but store the exact yardline for audit.

---

## Regime-labeling rules

Relabeling also assigns behavioral regime flags.

### `late_half_flag`
Set true when the possession begins in a half-ending state, typically with limited remaining time in Q2 or Q4-half context.

### `late_game_flag`
Set true when the possession begins in a late regulation state, typically Q4 with limited time remaining.

### `clock_kill_flag`
Set true when the offense is primarily attempting to reduce time rather than maximize scoring.

### `clock_kill_intensity`
Use:

- `0 = normal`
- `1 = mild bleed-clock`
- `2 = hard kill / kneel-out`

### `kneel_sequence_flag`
Set true when kneels occur or the possession clearly functions as a kneel-out sequence.

### `desperation_flag`
Set true when the offense is trailing and behaving with elevated urgency, hurry-up tempo, or must-score aggression.

---

## Conflict-resolution rules

When multiple candidate labels exist, use this priority:

1. touchdown return states override raw drive text
2. defensive and special-teams touchdowns override base turnover or kick labels
3. offensive touchdown overrides non-terminal intermediate events
4. end-of-half and end-of-game apply only when no higher-priority football event ends the possession first

Examples:

- interception on final play returned for touchdown -> `DEF_TD_ALLOWED`, not `END_HALF`
- blocked field goal returned for touchdown -> `ST_TD_ALLOWED`, not `FG_MISS`
- touchdown scored with no time left in half -> `OFF_TD`, not `END_HALF`

---

## Exclusion rules

Set `exclude_from_training = true` when any of the following occurs:

- unresolved terminal event
- broken or contradictory score change
- impossible clock transition
- next possession cannot be reconstructed reliably
- touchdown conversion cannot be resolved and affects scoring truth
- overtime possession appears in regulation table
- duplicate or corrupted drive identity
- drive text and play sequence conflict in a way that cannot be reconciled

Excluded possessions should remain in the processed dataset for audit and coverage reporting.

---

## QA requirements

A relabeled possession is trainable only if all of the following are true:

- exactly one canonical terminal state is assigned
- points reconcile with scoreboard change
- conversion fields are internally consistent
- next-state fields are internally consistent
- overtime status is correct
- required flags match the terminal state
- confidence is high enough for training inclusion

Recommended QA outputs:

- state frequency report
- exclusion-rate report
- state-by-state reconciliation report
- edge-case audit samples

---

## Modeling note

This relabeling system is designed to support the possession simulator that generates the pregame joint PMF over final score pairs:

\[
p(x,y)=P(\text{Team A}=x,\ \text{Team B}=y)
\]

Because that PMF is the foundation of fair odds, market comparison, and EV detection, relabeling accuracy is not a preprocessing detail. It is part of the model.

