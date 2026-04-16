from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "benchmark_name",
    "market_spread",
    "market_total",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare calibrated benchmark probabilities to market assumptions and output candidate bet tables."
    )
    parser.add_argument("--input", required=True, help="Path to calibrated benchmark eval CSV.")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for output files, e.g. outputs/benchmark_eval/fcs_benchmark_market_compare",
    )
    parser.add_argument(
        "--default-american-odds",
        type=float,
        default=-110.0,
        help="Default assumed price when explicit side odds are unavailable.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.02,
        help="Minimum model edge over market implied probability for candidate tables.",
    )
    parser.add_argument(
        "--min-ev",
        type=float,
        default=0.0,
        help="Minimum EV per 1 unit staked for candidate tables.",
    )
    parser.add_argument(
        "--home-spread-odds-col",
        default=None,
        help="Optional column containing offered American odds for home spread bet.",
    )
    parser.add_argument(
        "--away-spread-odds-col",
        default=None,
        help="Optional column containing offered American odds for away spread bet.",
    )
    parser.add_argument(
        "--over-odds-col",
        default=None,
        help="Optional column containing offered American odds for over bet.",
    )
    parser.add_argument(
        "--under-odds-col",
        default=None,
        help="Optional column containing offered American odds for under bet.",
    )
    return parser.parse_args()


def load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    df = pd.read_csv(p)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    return df


def choose_prob(df: pd.DataFrame, cal_col: str, raw_col: str) -> pd.Series:
    if cal_col in df.columns:
        cal = pd.to_numeric(df[cal_col], errors="coerce")
    else:
        cal = pd.Series(np.nan, index=df.index)

    if raw_col in df.columns:
        raw = pd.to_numeric(df[raw_col], errors="coerce")
    else:
        raw = pd.Series(np.nan, index=df.index)

    return cal.where(cal.notna(), raw)


def american_to_decimal(odds: float) -> float:
    if np.isnan(odds):
        return np.nan
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_implied_prob(odds: float) -> float:
    if np.isnan(odds):
        return np.nan
    dec = american_to_decimal(odds)
    return np.nan if np.isnan(dec) else 1.0 / dec


def ev_per_unit(prob: float, american_odds: float) -> float:
    if np.isnan(prob) or np.isnan(american_odds):
        return np.nan
    dec = american_to_decimal(american_odds)
    if np.isnan(dec):
        return np.nan
    win_profit = dec - 1.0
    lose_cost = 1.0
    return prob * win_profit - (1.0 - prob) * lose_cost


def has_quarter_line(x: float) -> bool:
    if pd.isna(x):
        return False
    frac = abs(x - round(x))
    frac = round(frac, 2)
    return frac in {0.25, 0.75}


def implied_prob_from_optional_col(df: pd.DataFrame, colname: str | None, default_odds: float) -> pd.Series:
    if colname is not None and colname in df.columns:
        odds = pd.to_numeric(df[colname], errors="coerce")
        out = odds.apply(american_to_implied_prob)
        default_prob = american_to_implied_prob(default_odds)
        return out.fillna(default_prob)
    return pd.Series(american_to_implied_prob(default_odds), index=df.index, dtype=float)


def odds_from_optional_col(df: pd.DataFrame, colname: str | None, default_odds: float) -> pd.Series:
    if colname is not None and colname in df.columns:
        odds = pd.to_numeric(df[colname], errors="coerce")
        return odds.fillna(default_odds)
    return pd.Series(default_odds, index=df.index, dtype=float)


def make_long_candidate_tables(df: pd.DataFrame, min_edge: float, min_ev: float) -> dict[str, pd.DataFrame]:
    home_spread = df.loc[
        (df["home_spread_edge"] >= min_edge) & (df["home_spread_ev_per_unit"] >= min_ev)
    ].copy()
    away_spread = df.loc[
        (df["away_spread_edge"] >= min_edge) & (df["away_spread_ev_per_unit"] >= min_ev)
    ].copy()
    over_total = df.loc[
        (df["over_total_edge"] >= min_edge) & (df["over_total_ev_per_unit"] >= min_ev)
    ].copy()
    under_total = df.loc[
        (df["under_total_edge"] >= min_edge) & (df["under_total_ev_per_unit"] >= min_ev)
    ].copy()

    home_spread = home_spread.sort_values(
        ["benchmark_name", "home_spread_ev_per_unit", "home_spread_edge"],
        ascending=[True, False, False],
    )
    away_spread = away_spread.sort_values(
        ["benchmark_name", "away_spread_ev_per_unit", "away_spread_edge"],
        ascending=[True, False, False],
    )
    over_total = over_total.sort_values(
        ["benchmark_name", "over_total_ev_per_unit", "over_total_edge"],
        ascending=[True, False, False],
    )
    under_total = under_total.sort_values(
        ["benchmark_name", "under_total_ev_per_unit", "under_total_edge"],
        ascending=[True, False, False],
    )

    return {
        "home_spread": home_spread,
        "away_spread": away_spread,
        "over_total": over_total,
        "under_total": under_total,
    }


def main() -> None:
    args = parse_args()
    df = load_df(args.input).copy()

    # Use calibrated probabilities when available; otherwise use raw benchmark probabilities.
    df["home_spread_model_prob"] = choose_prob(df, "home_spread_cover_prob_cal", "home_spread_cover_prob")
    df["over_total_model_prob"] = choose_prob(df, "over_total_prob_cal", "over_total_prob")
    df["home_ml_model_prob"] = choose_prob(df, "home_ml_ot_adj_prob_cal", "home_ml_ot_adj_prob")

    # Away/under derived as complements for comparison purposes.
    # This is acceptable here because we are using line-specific market assumptions,
    # and most recorded lines are non-integer medians without push mass.
    df["away_spread_model_prob"] = 1.0 - df["home_spread_model_prob"]
    df["under_total_model_prob"] = 1.0 - df["over_total_model_prob"]

    # Market implied probabilities. If odds columns are unavailable, assume default American odds.
    df["home_spread_market_prob"] = implied_prob_from_optional_col(
        df, args.home_spread_odds_col, args.default_american_odds
    )
    df["away_spread_market_prob"] = implied_prob_from_optional_col(
        df, args.away_spread_odds_col, args.default_american_odds
    )
    df["over_total_market_prob"] = implied_prob_from_optional_col(
        df, args.over_odds_col, args.default_american_odds
    )
    df["under_total_market_prob"] = implied_prob_from_optional_col(
        df, args.under_odds_col, args.default_american_odds
    )

    df["home_spread_offered_odds"] = odds_from_optional_col(
        df, args.home_spread_odds_col, args.default_american_odds
    )
    df["away_spread_offered_odds"] = odds_from_optional_col(
        df, args.away_spread_odds_col, args.default_american_odds
    )
    df["over_total_offered_odds"] = odds_from_optional_col(
        df, args.over_odds_col, args.default_american_odds
    )
    df["under_total_offered_odds"] = odds_from_optional_col(
        df, args.under_odds_col, args.default_american_odds
    )

    # Edge fields
    df["home_spread_edge"] = df["home_spread_model_prob"] - df["home_spread_market_prob"]
    df["away_spread_edge"] = df["away_spread_model_prob"] - df["away_spread_market_prob"]
    df["over_total_edge"] = df["over_total_model_prob"] - df["over_total_market_prob"]
    df["under_total_edge"] = df["under_total_model_prob"] - df["under_total_market_prob"]

    # EV per 1 unit staked
    df["home_spread_ev_per_unit"] = [
        ev_per_unit(p, o) for p, o in zip(df["home_spread_model_prob"], df["home_spread_offered_odds"])
    ]
    df["away_spread_ev_per_unit"] = [
        ev_per_unit(p, o) for p, o in zip(df["away_spread_model_prob"], df["away_spread_offered_odds"])
    ]
    df["over_total_ev_per_unit"] = [
        ev_per_unit(p, o) for p, o in zip(df["over_total_model_prob"], df["over_total_offered_odds"])
    ]
    df["under_total_ev_per_unit"] = [
        ev_per_unit(p, o) for p, o in zip(df["under_total_model_prob"], df["under_total_offered_odds"])
    ]

    # Line-shape flags
    df["quarter_spread_flag"] = pd.to_numeric(df["market_spread"], errors="coerce").apply(has_quarter_line)
    df["quarter_total_flag"] = pd.to_numeric(df["market_total"], errors="coerce").apply(has_quarter_line)

    # Summary by benchmark
    summary_rows = []
    for benchmark_name, g in df.groupby("benchmark_name", sort=True):
        summary_rows.append(
            {
                "benchmark_name": benchmark_name,
                "n_rows": int(len(g)),
                "mean_home_spread_edge": float(g["home_spread_edge"].mean()),
                "mean_away_spread_edge": float(g["away_spread_edge"].mean()),
                "mean_over_total_edge": float(g["over_total_edge"].mean()),
                "mean_under_total_edge": float(g["under_total_edge"].mean()),
                "mean_home_spread_ev_per_unit": float(g["home_spread_ev_per_unit"].mean()),
                "mean_away_spread_ev_per_unit": float(g["away_spread_ev_per_unit"].mean()),
                "mean_over_total_ev_per_unit": float(g["over_total_ev_per_unit"].mean()),
                "mean_under_total_ev_per_unit": float(g["under_total_ev_per_unit"].mean()),
                "quarter_spread_rate": float(g["quarter_spread_flag"].mean()),
                "quarter_total_rate": float(g["quarter_total_flag"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    candidate_tables = make_long_candidate_tables(df, min_edge=args.min_edge, min_ev=args.min_ev)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    full_eval_path = output_prefix.with_name(output_prefix.name + "_full_eval.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    home_spread_path = output_prefix.with_name(output_prefix.name + "_home_spread_candidates.csv")
    away_spread_path = output_prefix.with_name(output_prefix.name + "_away_spread_candidates.csv")
    over_total_path = output_prefix.with_name(output_prefix.name + "_over_total_candidates.csv")
    under_total_path = output_prefix.with_name(output_prefix.name + "_under_total_candidates.csv")

    df.to_csv(full_eval_path, index=False)
    summary.to_csv(summary_path, index=False)
    candidate_tables["home_spread"].to_csv(home_spread_path, index=False)
    candidate_tables["away_spread"].to_csv(away_spread_path, index=False)
    candidate_tables["over_total"].to_csv(over_total_path, index=False)
    candidate_tables["under_total"].to_csv(under_total_path, index=False)

    print(f"Wrote full comparison table to {full_eval_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote home spread candidates to {home_spread_path}")
    print(f"Wrote away spread candidates to {away_spread_path}")
    print(f"Wrote over total candidates to {over_total_path}")
    print(f"Wrote under total candidates to {under_total_path}")

    print("\nSummary:")
    if not summary.empty:
        print(summary.to_string(index=False))

    print("\nCandidate counts:")
    print({
        "home_spread": int(len(candidate_tables["home_spread"])),
        "away_spread": int(len(candidate_tables["away_spread"])),
        "over_total": int(len(candidate_tables["over_total"])),
        "under_total": int(len(candidate_tables["under_total"])),
    })


if __name__ == "__main__":
    main()
