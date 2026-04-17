from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest reference-model candidate bets on historical outcomes."
    )
    parser.add_argument("--input", required=True, help="Path to reference market compare full eval CSV.")
    parser.add_argument("--output-prefix", required=True, help="Prefix for output files.")
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--min-ev", type=float, default=0.0)
    return parser.parse_args()


def load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    return pd.read_csv(p)


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def realized_profit(odds: float, won: float) -> float:
    dec = american_to_decimal(float(odds))
    return (dec - 1.0) if float(won) == 1.0 else -1.0


def build_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df.iterrows():
        rows.append(
            {
                "market_type": "home_spread",
                "season": r.get("season"),
                "week": r.get("week"),
                "start_date": r.get("start_date"),
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "line": r.get("market_spread"),
                "model_prob": r.get("home_spread_ref_prob"),
                "market_prob": r.get("home_spread_market_prob"),
                "edge": r.get("home_spread_edge"),
                "ev_per_unit": r.get("home_spread_ev_per_unit"),
                "offered_odds": r.get("home_spread_offered_odds"),
                "won": r.get("actual_home_cover"),
            }
        )
        rows.append(
            {
                "market_type": "away_spread",
                "season": r.get("season"),
                "week": r.get("week"),
                "start_date": r.get("start_date"),
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "line": r.get("market_spread"),
                "model_prob": r.get("away_spread_ref_prob"),
                "market_prob": r.get("away_spread_market_prob"),
                "edge": r.get("away_spread_edge"),
                "ev_per_unit": r.get("away_spread_ev_per_unit"),
                "offered_odds": r.get("away_spread_offered_odds"),
                "won": 1.0 - float(r.get("actual_home_cover")),
            }
        )
        rows.append(
            {
                "market_type": "over_total",
                "season": r.get("season"),
                "week": r.get("week"),
                "start_date": r.get("start_date"),
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "line": r.get("market_total"),
                "model_prob": r.get("over_total_ref_prob"),
                "market_prob": r.get("over_total_market_prob"),
                "edge": r.get("over_total_edge"),
                "ev_per_unit": r.get("over_total_ev_per_unit"),
                "offered_odds": r.get("over_total_offered_odds"),
                "won": r.get("actual_over_total"),
            }
        )
        rows.append(
            {
                "market_type": "under_total",
                "season": r.get("season"),
                "week": r.get("week"),
                "start_date": r.get("start_date"),
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "line": r.get("market_total"),
                "model_prob": r.get("under_total_ref_prob"),
                "market_prob": r.get("under_total_market_prob"),
                "edge": r.get("under_total_edge"),
                "ev_per_unit": r.get("under_total_ev_per_unit"),
                "offered_odds": r.get("under_total_offered_odds"),
                "won": 1.0 - float(r.get("actual_over_total")),
            }
        )

    out = pd.DataFrame(rows)
    out["realized_profit_per_bet"] = [
        realized_profit(o, w) for o, w in zip(out["offered_odds"], out["won"])
    ]
    return out


def main() -> None:
    args = parse_args()
    df = load_df(args.input)
    long_df = build_long(df)

    candidates = long_df.loc[
        (long_df["edge"] >= args.min_edge) & (long_df["ev_per_unit"] >= args.min_ev)
    ].copy()

    summary = (
        candidates.groupby("market_type", sort=True)
        .agg(
            n_bets=("market_type", "size"),
            mean_model_prob=("model_prob", "mean"),
            mean_market_prob=("market_prob", "mean"),
            mean_edge=("edge", "mean"),
            mean_ev_per_unit=("ev_per_unit", "mean"),
            hit_rate=("won", "mean"),
            realized_roi_per_bet=("realized_profit_per_bet", "mean"),
            total_profit_units=("realized_profit_per_bet", "sum"),
        )
        .reset_index()
    )

    season_summary = (
        candidates.groupby(["market_type", "season"], sort=True)
        .agg(
            n_bets=("market_type", "size"),
            hit_rate=("won", "mean"),
            realized_roi_per_bet=("realized_profit_per_bet", "mean"),
            total_profit_units=("realized_profit_per_bet", "sum"),
        )
        .reset_index()
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(output_prefix.with_name(output_prefix.name + "_candidates.csv"), index=False)
    summary.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    season_summary.to_csv(output_prefix.with_name(output_prefix.name + "_season_summary.csv"), index=False)

    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("No candidates met the thresholds.")


if __name__ == "__main__":
    main()
