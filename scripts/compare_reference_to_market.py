from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "home_spread_ref_prob",
    "away_spread_ref_prob",
    "over_total_ref_prob",
    "under_total_ref_prob",
    "market_spread",
    "market_total",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reference model probabilities to market assumptions and create candidate bet tables."
    )
    parser.add_argument("--input", required=True, help="Path to reference eval CSV.")
    parser.add_argument("--output-prefix", required=True, help="Prefix for output files.")
    parser.add_argument("--default-american-odds", type=float, default=-110.0)
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--min-ev", type=float, default=0.0)
    return parser.parse_args()


def load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    df = pd.read_csv(p)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {sorted(missing)}")
    return df


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_implied_prob(odds: float) -> float:
    return 1.0 / american_to_decimal(odds)


def ev_per_unit(prob: float, odds: float) -> float:
    dec = american_to_decimal(odds)
    return prob * (dec - 1.0) - (1.0 - prob)


def has_quarter_line(x: float) -> bool:
    if pd.isna(x):
        return False
    frac = round(abs(x - round(x)), 2)
    return frac in {0.25, 0.75}


def main() -> None:
    args = parse_args()
    df = load_df(args.input).copy()

    market_prob = american_to_implied_prob(args.default_american_odds)

    df["home_spread_market_prob"] = market_prob
    df["away_spread_market_prob"] = market_prob
    df["over_total_market_prob"] = market_prob
    df["under_total_market_prob"] = market_prob

    df["home_spread_offered_odds"] = args.default_american_odds
    df["away_spread_offered_odds"] = args.default_american_odds
    df["over_total_offered_odds"] = args.default_american_odds
    df["under_total_offered_odds"] = args.default_american_odds

    df["home_spread_edge"] = df["home_spread_ref_prob"] - df["home_spread_market_prob"]
    df["away_spread_edge"] = df["away_spread_ref_prob"] - df["away_spread_market_prob"]
    df["over_total_edge"] = df["over_total_ref_prob"] - df["over_total_market_prob"]
    df["under_total_edge"] = df["under_total_ref_prob"] - df["under_total_market_prob"]

    df["home_spread_ev_per_unit"] = df["home_spread_ref_prob"].apply(lambda p: ev_per_unit(float(p), args.default_american_odds))
    df["away_spread_ev_per_unit"] = df["away_spread_ref_prob"].apply(lambda p: ev_per_unit(float(p), args.default_american_odds))
    df["over_total_ev_per_unit"] = df["over_total_ref_prob"].apply(lambda p: ev_per_unit(float(p), args.default_american_odds))
    df["under_total_ev_per_unit"] = df["under_total_ref_prob"].apply(lambda p: ev_per_unit(float(p), args.default_american_odds))

    df["quarter_spread_flag"] = pd.to_numeric(df["market_spread"], errors="coerce").apply(has_quarter_line)
    df["quarter_total_flag"] = pd.to_numeric(df["market_total"], errors="coerce").apply(has_quarter_line)

    summary = pd.DataFrame(
        [
            {
                "market_type": "home_spread",
                "n_rows": int(len(df)),
                "mean_model_prob": float(df["home_spread_ref_prob"].mean()),
                "mean_market_prob": float(df["home_spread_market_prob"].mean()),
                "mean_edge": float(df["home_spread_edge"].mean()),
                "mean_ev_per_unit": float(df["home_spread_ev_per_unit"].mean()),
            },
            {
                "market_type": "away_spread",
                "n_rows": int(len(df)),
                "mean_model_prob": float(df["away_spread_ref_prob"].mean()),
                "mean_market_prob": float(df["away_spread_market_prob"].mean()),
                "mean_edge": float(df["away_spread_edge"].mean()),
                "mean_ev_per_unit": float(df["away_spread_ev_per_unit"].mean()),
            },
            {
                "market_type": "over_total",
                "n_rows": int(len(df)),
                "mean_model_prob": float(df["over_total_ref_prob"].mean()),
                "mean_market_prob": float(df["over_total_market_prob"].mean()),
                "mean_edge": float(df["over_total_edge"].mean()),
                "mean_ev_per_unit": float(df["over_total_ev_per_unit"].mean()),
            },
            {
                "market_type": "under_total",
                "n_rows": int(len(df)),
                "mean_model_prob": float(df["under_total_ref_prob"].mean()),
                "mean_market_prob": float(df["under_total_market_prob"].mean()),
                "mean_edge": float(df["under_total_edge"].mean()),
                "mean_ev_per_unit": float(df["under_total_ev_per_unit"].mean()),
            },
        ]
    )

    home_spread = df.loc[(df["home_spread_edge"] >= args.min_edge) & (df["home_spread_ev_per_unit"] >= args.min_ev)].copy()
    away_spread = df.loc[(df["away_spread_edge"] >= args.min_edge) & (df["away_spread_ev_per_unit"] >= args.min_ev)].copy()
    over_total = df.loc[(df["over_total_edge"] >= args.min_edge) & (df["over_total_ev_per_unit"] >= args.min_ev)].copy()
    under_total = df.loc[(df["under_total_edge"] >= args.min_edge) & (df["under_total_ev_per_unit"] >= args.min_ev)].copy()

    home_spread = home_spread.sort_values(["home_spread_ev_per_unit", "home_spread_edge"], ascending=False)
    away_spread = away_spread.sort_values(["away_spread_ev_per_unit", "away_spread_edge"], ascending=False)
    over_total = over_total.sort_values(["over_total_ev_per_unit", "over_total_edge"], ascending=False)
    under_total = under_total.sort_values(["under_total_ev_per_unit", "under_total_edge"], ascending=False)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_prefix.with_name(output_prefix.name + "_full_eval.csv"), index=False)
    summary.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    home_spread.to_csv(output_prefix.with_name(output_prefix.name + "_home_spread_candidates.csv"), index=False)
    away_spread.to_csv(output_prefix.with_name(output_prefix.name + "_away_spread_candidates.csv"), index=False)
    over_total.to_csv(output_prefix.with_name(output_prefix.name + "_over_total_candidates.csv"), index=False)
    under_total.to_csv(output_prefix.with_name(output_prefix.name + "_under_total_candidates.csv"), index=False)

    print(summary.to_string(index=False))
    print(
        {
            "home_spread_candidates": int(len(home_spread)),
            "away_spread_candidates": int(len(away_spread)),
            "over_total_candidates": int(len(over_total)),
            "under_total_candidates": int(len(under_total)),
        }
    )


if __name__ == "__main__":
    main()
