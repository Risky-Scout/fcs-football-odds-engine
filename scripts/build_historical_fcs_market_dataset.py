from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {
    "Authorization": f"Bearer {os.environ.get('CFBD_API_KEY', '')}",
    "Accept": "application/json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build historical FCS dataset with market prices from CFBD."
    )
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    parser.add_argument("--season-type", default="regular", choices=["regular", "postseason", "both"])
    parser.add_argument("--raw-dir", default="data/raw/cfbd")
    parser.add_argument("--out-csv", default="data/processed/fcs_historical_games_market.csv")
    parser.add_argument("--out-parquet", default="data/processed/fcs_historical_games_market.parquet")
    return parser.parse_args()


def require_api_key() -> None:
    if not os.environ.get("CFBD_API_KEY"):
        raise RuntimeError("CFBD_API_KEY is not set in the environment.")


def get_json(endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    resp = requests.get(
        f"{BASE_URL}/{endpoint}",
        headers=HEADERS,
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Expected list response from {endpoint}, got {type(data)}")
    return data


def flatten_game_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).copy()

    rename_map = {}
    if "id" in df.columns and "game_id" not in df.columns:
        rename_map["id"] = "game_id"
    if rename_map:
        df = df.rename(columns=rename_map)

    keep_cols = [
        c for c in [
            "game_id",
            "season",
            "week",
            "season_type",
            "start_date",
            "start_time_tbd",
            "completed",
            "neutral_site",
            "conference_game",
            "attendance",
            "venue_id",
            "venue",
            "home_id",
            "home_team",
            "home_conference",
            "home_classification",
            "home_points",
            "away_id",
            "away_team",
            "away_conference",
            "away_classification",
            "away_points",
            "excitement_index",
            "notes",
        ]
        if c in df.columns
    ]
    return df[keep_cols].copy()


def extract_consensus_from_lines(lines_obj: Any) -> dict[str, Any]:
    """
    CFBD lines are typically nested per provider.
    We aggregate to a simple consensus using medians when available.
    """
    result = {
        "line_providers_n": 0,
        "market_spread": np.nan,
        "market_spread_open": np.nan,
        "market_total": np.nan,
        "market_total_open": np.nan,
        "home_moneyline": np.nan,
        "away_moneyline": np.nan,
    }

    if not isinstance(lines_obj, list) or len(lines_obj) == 0:
        return result

    providers = []
    spreads = []
    spreads_open = []
    totals = []
    totals_open = []
    home_ml = []
    away_ml = []

    for item in lines_obj:
        if not isinstance(item, dict):
            continue
        providers.append(item.get("provider"))
        for key, bucket in [
            ("spread", spreads),
            ("spreadOpen", spreads_open),
            ("overUnder", totals),
            ("overUnderOpen", totals_open),
            ("homeMoneyline", home_ml),
            ("awayMoneyline", away_ml),
        ]:
            val = item.get(key)
            if val is not None:
                try:
                    bucket.append(float(val))
                except (TypeError, ValueError):
                    pass

    result["line_providers_n"] = len([p for p in providers if p is not None])
    if spreads:
        result["market_spread"] = float(np.median(spreads))
    if spreads_open:
        result["market_spread_open"] = float(np.median(spreads_open))
    if totals:
        result["market_total"] = float(np.median(totals))
    if totals_open:
        result["market_total_open"] = float(np.median(totals_open))
    if home_ml:
        result["home_moneyline"] = float(np.median(home_ml))
    if away_ml:
        result["away_moneyline"] = float(np.median(away_ml))

    return result


def flatten_lines_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    flat_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        game_id = row.get("gameId", row.get("id", row.get("game_id")))
        consensus = extract_consensus_from_lines(row.get("lines", []))

        flat = {
            "game_id": game_id,
            "season": row.get("season"),
            "week": row.get("week"),
            "season_type": row.get("seasonType", row.get("season_type")),
            "start_date": row.get("startDate", row.get("start_date")),
            "home_team": row.get("homeTeam", row.get("home_team")),
            "away_team": row.get("awayTeam", row.get("away_team")),
            "home_conference": row.get("homeConference", row.get("home_conference")),
            "away_conference": row.get("awayConference", row.get("away_conference")),
            "home_classification": row.get("homeClassification", row.get("home_classification")),
            "away_classification": row.get("awayClassification", row.get("away_classification")),
            **consensus,
        }
        flat_rows.append(flat)

    return pd.DataFrame(flat_rows)


def load_season_data(season: int, season_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    games_rows = get_json("games", {"year": season, "seasonType": season_type})
    lines_rows = get_json("lines", {"year": season, "seasonType": season_type})

    games_df = flatten_game_rows(games_rows)
    lines_df = flatten_lines_rows(lines_rows)
    return games_df, lines_df


def filter_fcs_with_prices(df: pd.DataFrame) -> pd.DataFrame:
    # Keep games where at least one team is classified as FCS, and where market prices exist.
    out = df.copy()

    home_cls = out["home_classification"].astype(str).str.lower() if "home_classification" in out.columns else pd.Series("", index=out.index)
    away_cls = out["away_classification"].astype(str).str.lower() if "away_classification" in out.columns else pd.Series("", index=out.index)

    is_fcs = home_cls.eq("fcs") | away_cls.eq("fcs")
    has_market = (
        out["market_spread"].notna()
        | out["market_total"].notna()
        | out["home_moneyline"].notna()
        | out["away_moneyline"].notna()
    )

    out = out.loc[is_fcs & has_market].copy()
    return out


def main() -> None:
    require_api_key()
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_games = []
    all_lines = []

    for season in range(args.start_season, args.end_season + 1):
        print(f"Downloading season {season}...")
        games_df, lines_df = load_season_data(season, args.season_type)

        games_path = raw_dir / f"games_{season}_{args.season_type}.csv"
        lines_path = raw_dir / f"lines_{season}_{args.season_type}.csv"

        games_df.to_csv(games_path, index=False)
        lines_df.to_csv(lines_path, index=False)

        all_games.append(games_df)
        all_lines.append(lines_df)

    games = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()
    lines = pd.concat(all_lines, ignore_index=True) if all_lines else pd.DataFrame()

    if games.empty or lines.empty:
        raise RuntimeError("No games or lines were downloaded.")

    merged = games.merge(
        lines,
        on=["game_id"],
        how="inner",
        suffixes=("", "_line"),
    )

    # Prefer game table values when present, otherwise line table values.
    for col in ["season", "week", "season_type", "start_date", "home_team", "away_team",
                "home_conference", "away_conference", "home_classification", "away_classification"]:
        line_col = f"{col}_line"
        if line_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[line_col])

    merged["home_score"] = merged["home_points"] if "home_points" in merged.columns else np.nan
    merged["away_score"] = merged["away_points"] if "away_points" in merged.columns else np.nan

    final_cols = [
        c for c in [
            "game_id",
            "season",
            "week",
            "season_type",
            "start_date",
            "neutral_site",
            "conference_game",
            "home_team",
            "away_team",
            "home_conference",
            "away_conference",
            "home_classification",
            "away_classification",
            "home_score",
            "away_score",
            "market_spread",
            "market_spread_open",
            "market_total",
            "market_total_open",
            "home_moneyline",
            "away_moneyline",
            "line_providers_n",
        ]
        if c in merged.columns
    ]

    final_df = merged[final_cols].copy()
    final_df = filter_fcs_with_prices(final_df)
    final_df = final_df.sort_values(["season", "week", "start_date", "game_id"]).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_parquet = Path(args.out_parquet)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(out_csv, index=False)
    final_df.to_parquet(out_parquet, index=False)

    print(f"Wrote {len(final_df)} rows to {out_csv}")
    print(f"Wrote {len(final_df)} rows to {out_parquet}")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
