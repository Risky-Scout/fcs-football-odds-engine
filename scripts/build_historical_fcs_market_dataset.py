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
        description="Build historical FCS-vs-FCS dataset with market prices from CFBD."
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


def rename_common_game_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "id": "game_id",
        "seasonType": "season_type",
        "startDate": "start_date",
        "startTimeTBD": "start_time_tbd",
        "neutralSite": "neutral_site",
        "conferenceGame": "conference_game",
        "venueId": "venue_id",
        "homeId": "home_id",
        "homeTeam": "home_team",
        "homeConference": "home_conference",
        "homeClassification": "home_classification",
        "homePoints": "home_points",
        "awayId": "away_id",
        "awayTeam": "away_team",
        "awayConference": "away_conference",
        "awayClassification": "away_classification",
        "awayPoints": "away_points",
        "excitementIndex": "excitement_index",
    }
    usable = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
    if usable:
        df = df.rename(columns=usable)
    return df


def rename_common_line_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "id": "game_id",
        "gameId": "game_id",
        "seasonType": "season_type",
        "startDate": "start_date",
        "homeTeam": "home_team",
        "awayTeam": "away_team",
        "homeConference": "home_conference",
        "awayConference": "away_conference",
        "homeClassification": "home_classification",
        "awayClassification": "away_classification",
    }
    usable = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
    if usable:
        df = df.rename(columns=usable)
    return df



def normalize_game_id_col(df: pd.DataFrame) -> pd.DataFrame:
    if "game_id" not in df.columns:
        return df
    out = df.copy()
    out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
    out = out.loc[out["game_id"].notna()].copy()
    return out


def flatten_game_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).copy()
    df = rename_common_game_cols(df)

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

        row_df = pd.DataFrame([row])
        row_df = rename_common_line_cols(row_df)
        row2 = row_df.iloc[0].to_dict()

        game_id = row2.get("game_id")
        if game_id is None or (isinstance(game_id, float) and pd.isna(game_id)):
            game_id = row.get("gameId", row.get("id"))

        flat = {
            "game_id": game_id,
            "season": row2.get("season"),
            "week": row2.get("week"),
            "season_type": row2.get("season_type"),
            "start_date": row2.get("start_date"),
            "home_team": row2.get("home_team"),
            "away_team": row2.get("away_team"),
            "home_conference": row2.get("home_conference"),
            "away_conference": row2.get("away_conference"),
            "home_classification": row2.get("home_classification"),
            "away_classification": row2.get("away_classification"),
            **extract_consensus_from_lines(row.get("lines", [])),
        }
        flat_rows.append(flat)

    return pd.DataFrame(flat_rows)


def load_season_data(season: int, season_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    games_rows = get_json("games", {"year": season, "seasonType": season_type})
    lines_rows = get_json("lines", {"year": season, "seasonType": season_type})

    games_df = flatten_game_rows(games_rows)
    lines_df = flatten_lines_rows(lines_rows)
    return games_df, lines_df


def filter_fcs_vs_fcs_with_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    home_cls = out["home_classification"].astype(str).str.lower()
    away_cls = out["away_classification"].astype(str).str.lower()

    is_fcs_vs_fcs = home_cls.eq("fcs") & away_cls.eq("fcs")
    has_spread_or_total = out["market_spread"].notna() | out["market_total"].notna()
    has_scores = out["home_score"].notna() & out["away_score"].notna()

    out = out.loc[is_fcs_vs_fcs & has_spread_or_total & has_scores].copy()
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

        games_df.to_csv(raw_dir / f"games_{season}_{args.season_type}.csv", index=False)
        lines_df.to_csv(raw_dir / f"lines_{season}_{args.season_type}.csv", index=False)

        all_games.append(games_df)
        all_lines.append(lines_df)

    games = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()
    lines = pd.concat(all_lines, ignore_index=True) if all_lines else pd.DataFrame()

    if games.empty or lines.empty:
        raise RuntimeError("No games or lines were downloaded.")

    games = normalize_game_id_col(games)
    lines = normalize_game_id_col(lines)

    merged = games.merge(lines, on=["game_id"], how="inner", suffixes=("", "_line"))

    for col in [
        "season", "week", "season_type", "start_date",
        "home_team", "away_team", "home_conference", "away_conference",
        "home_classification", "away_classification"
    ]:
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
    final_df = filter_fcs_vs_fcs_with_prices(final_df)
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
    print("\nMissingness:")
    print(final_df[["home_score", "away_score", "market_spread", "market_total", "home_moneyline", "away_moneyline"]].isna().mean())


if __name__ == "__main__":
    main()
