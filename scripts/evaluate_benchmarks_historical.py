from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks import benchmark_game_probs, run_massey_benchmark_for_game, run_pi_benchmark_for_game  # noqa: E402


REQUIRED_COLS = {"home_team", "away_team", "home_score", "away_score"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Historical benchmark evaluation for FCS benchmark models."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV or Parquet file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument(
        "--min-train-games",
        type=int,
        default=20,
        help="Minimum number of prior games required before scoring held-out games.",
    )
    parser.add_argument(
        "--spread-col",
        default="market_spread",
        help="Column name for market spread. Optional.",
    )
    parser.add_argument(
        "--total-col",
        default="market_total",
        help="Column name for market total. Optional.",
    )
    parser.add_argument(
        "--home-team-total-col",
        default="home_team_total",
        help="Column name for home team total. Optional.",
    )
    parser.add_argument(
        "--away-team-total-col",
        default="away_team_total",
        help="Column name for away team total. Optional.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=10.0,
        help="Ridge penalty for Massey benchmark.",
    )
    parser.add_argument(
        "--recency-halflife-weeks",
        type=float,
        default=2.0,
        help="Recency half-life in weeks for Massey benchmark.",
    )
    parser.add_argument(
        "--rho-offense",
        type=float,
        default=0.90,
        help="Pi offense persistence.",
    )
    parser.add_argument(
        "--rho-defense",
        type=float,
        default=0.90,
        help="Pi defense persistence.",
    )
    parser.add_argument(
        "--k-offense",
        type=float,
        default=0.10,
        help="Pi offense update gain.",
    )
    parser.add_argument(
        "--k-defense",
        type=float,
        default=0.10,
        help="Pi defense update gain.",
    )
    return parser.parse_args()


def load_games(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        raise ValueError("Input must be CSV or Parquet")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    return df


def sort_games(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["season", "week", "game_date"] if c in df.columns]
    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)
    return df.reset_index(drop=True)


def safe_val(row: pd.Series, col: str):
    if col in row.index and pd.notna(row[col]):
        return row[col]
    return None


def actual_home_win(row: pd.Series) -> int:
    return int(float(row["home_score"]) > float(row["away_score"]))


def actual_home_cover(row: pd.Series, spread: float | None) -> float | None:
    if spread is None:
        return None
    return float((float(row["home_score"]) - float(row["away_score"])) > float(spread))


def actual_over(row: pd.Series, total: float | None) -> float | None:
    if total is None:
        return None
    return float((float(row["home_score"]) + float(row["away_score"])) > float(total))


def build_row(
    benchmark_name: str,
    result,
    probs: dict,
    heldout_row: pd.Series,
    spread: float | None,
    total: float | None,
    home_team_total: float | None,
    away_team_total: float | None,
) -> dict:
    out = {
        "benchmark_name": benchmark_name,
        "home_team": heldout_row["home_team"],
        "away_team": heldout_row["away_team"],
        "actual_home_score": float(heldout_row["home_score"]),
        "actual_away_score": float(heldout_row["away_score"]),
        "expected_home_score": result.expected_home_score,
        "expected_away_score": result.expected_away_score,
        "expected_margin_home": result.expected_margin_home,
        "expected_total": result.expected_total,
        "home_ml_prob": probs.get("home_ml_prob"),
        "away_ml_prob": probs.get("away_ml_prob"),
        "tie_prob": probs.get("tie_prob"),
        "home_ml_ot_adj_prob": probs.get("home_ml_ot_adj_prob"),
        "away_ml_ot_adj_prob": probs.get("away_ml_ot_adj_prob"),
        "actual_home_win": actual_home_win(heldout_row),
        "market_spread": spread,
        "market_total": total,
        "home_team_total_line": home_team_total,
        "away_team_total_line": away_team_total,
        "actual_home_cover": actual_home_cover(heldout_row, spread),
        "actual_over_total": actual_over(heldout_row, total),
    }

    if spread is not None:
        out["home_spread_cover_prob"] = probs.get(f"home_minus_{spread}_cover_prob")
        out["away_spread_cover_prob"] = probs.get(f"away_plus_{spread}_cover_prob")

    if total is not None:
        out["over_total_prob"] = probs.get(f"over_{total}_prob")
        out["under_total_prob"] = probs.get(f"under_{total}_prob")

    if home_team_total is not None:
        out["home_team_total_over_prob"] = probs.get(
            f"home_team_total_over_{home_team_total}_prob"
        )
        out["home_team_total_under_prob"] = probs.get(
            f"home_team_total_under_{home_team_total}_prob"
        )

    if away_team_total is not None:
        out["away_team_total_over_prob"] = probs.get(
            f"away_team_total_over_{away_team_total}_prob"
        )
        out["away_team_total_under_prob"] = probs.get(
            f"away_team_total_under_{away_team_total}_prob"
        )

    for col in ["season", "week", "game_date", "neutral_site"]:
        if col in heldout_row.index:
            out[col] = heldout_row[col]

    return out


def main() -> None:
    args = parse_args()
    games = sort_games(load_games(args.input))

    if len(games) <= args.min_train_games:
        raise ValueError("Not enough games for held-out evaluation")

    rows: list[dict] = []

    for i in range(args.min_train_games, len(games)):
        train_df = games.iloc[:i].copy()
        heldout = games.iloc[i]

        home_team = heldout["home_team"]
        away_team = heldout["away_team"]
        neutral_site = bool(heldout["neutral_site"]) if "neutral_site" in heldout.index and pd.notna(heldout["neutral_site"]) else False

        spread = safe_val(heldout, args.spread_col)
        total = safe_val(heldout, args.total_col)
        home_team_total = safe_val(heldout, args.home_team_total_col)
        away_team_total = safe_val(heldout, args.away_team_total_col)

        massey_result = run_massey_benchmark_for_game(
            games_df=train_df,
            home_team=home_team,
            away_team=away_team,
            neutral_site=neutral_site,
            ridge=args.ridge,
            recency_halflife_weeks=args.recency_halflife_weeks,
        )
        massey_probs = benchmark_game_probs(
            massey_result,
            spread=spread,
            total=total,
            home_team_total=home_team_total,
            away_team_total=away_team_total,
        )
        rows.append(
            build_row(
                "massey",
                massey_result,
                massey_probs,
                heldout,
                spread,
                total,
                home_team_total,
                away_team_total,
            )
        )

        pi_result = run_pi_benchmark_for_game(
            games_df=train_df,
            home_team=home_team,
            away_team=away_team,
            neutral_site=neutral_site,
            rho_offense=args.rho_offense,
            rho_defense=args.rho_defense,
            k_offense=args.k_offense,
            k_defense=args.k_defense,
        )
        pi_probs = benchmark_game_probs(
            pi_result,
            spread=spread,
            total=total,
            home_team_total=home_team_total,
            away_team_total=away_team_total,
        )
        rows.append(
            build_row(
                "pi",
                pi_result,
                pi_probs,
                heldout,
                spread,
                total,
                home_team_total,
                away_team_total,
            )
        )

    out_df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Wrote {len(out_df)} benchmark rows to {output_path}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
