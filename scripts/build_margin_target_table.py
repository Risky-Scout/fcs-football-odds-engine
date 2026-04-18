from __future__ import annotations

import argparse
import ast
import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.collegefootballdata.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build historical halftime / second-half / full-game score and margin target table for the FCS PMF engine."
    )
    parser.add_argument(
        "--market-input",
        default="data/processed/fcs_historical_games_market.csv",
        help="Processed FCS market dataset.",
    )
    parser.add_argument(
        "--season-type",
        default="regular",
        choices=["regular", "postseason", "both"],
        help="Season type for detailed CFBD games fetch.",
    )
    parser.add_argument(
        "--detail-dir",
        default="data/raw/cfbd_detail",
        help="Directory to cache detailed CFBD games with line scores.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/fcs_margin_targets.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/processed/fcs_margin_targets.parquet",
        help="Output Parquet path.",
    )
    return parser.parse_args()


def require_api_key_if_needed() -> None:
    if not os.environ.get("CFBD_API_KEY"):
        raise RuntimeError("CFBD_API_KEY is not set in the environment.")


def get_json(endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {os.environ.get('CFBD_API_KEY', '')}",
        "Accept": "application/json",
    }
    resp = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=60)
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
        "neutralSite": "neutral_site",
        "conferenceGame": "conference_game",
        "homeId": "home_id",
        "homeTeam": "home_team",
        "homeConference": "home_conference",
        "homeClassification": "home_classification",
        "homePoints": "home_points",
        "homeLineScores": "home_line_scores",
        "awayId": "away_id",
        "awayTeam": "away_team",
        "awayConference": "away_conference",
        "awayClassification": "away_classification",
        "awayPoints": "away_points",
        "awayLineScores": "away_line_scores",
    }
    usable = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
    if usable:
        df = df.rename(columns=usable)
    return df


def fetch_or_load_detailed_games(season: int, season_type: str, detail_dir: Path) -> pd.DataFrame:
    detail_dir.mkdir(parents=True, exist_ok=True)
    path = detail_dir / f"games_detailed_{season}_{season_type}.csv"

    if path.exists():
        df = pd.read_csv(path)
        return rename_common_game_cols(df)

    require_api_key_if_needed()
    rows = get_json("games", {"year": season, "seasonType": season_type})
    df = pd.DataFrame(rows).copy()
    df = rename_common_game_cols(df)
    df.to_csv(path, index=False)
    return df


def parse_line_scores(value: Any) -> list[int] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, list):
        out = []
        for x in value:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                return None
        return out

    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() == "nan":
            return None

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                out = []
                for x in parsed:
                    try:
                        out.append(int(x))
                    except (TypeError, ValueError):
                        return None
                return out
        except Exception:
            pass

        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            out = []
            for p in parts:
                if p == "":
                    continue
                try:
                    out.append(int(float(p)))
                except (TypeError, ValueError):
                    return None
            return out if out else None

    return None


def normalize_game_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
    out = out.loc[out["game_id"].notna()].copy()
    return out


def extract_segment_points(scores: list[int] | None, official_final: float | int | None) -> dict[str, float | bool]:
    out = {
        "line_scores_parsed_ok": False,
        "n_periods": np.nan,
        "points_1h": np.nan,
        "points_2h_reg": np.nan,
        "points_ot": np.nan,
        "points_2h": np.nan,
        "points_game_linesum": np.nan,
        "points_game_official": np.nan if official_final is None else float(official_final),
        "segment_identity_ok": False,
        "has_four_quarters": False,
    }

    if scores is None:
        return out

    n = len(scores)
    if n == 0:
        return out

    q1 = scores[0] if n >= 1 else 0
    q2 = scores[1] if n >= 2 else 0
    q3 = scores[2] if n >= 3 else 0
    q4 = scores[3] if n >= 4 else 0
    ot = sum(scores[4:]) if n >= 5 else 0

    points_1h = q1 + q2
    points_2h_reg = q3 + q4
    points_game_linesum = sum(scores)
    points_game_official = np.nan if official_final is None or pd.isna(official_final) else float(official_final)
    points_2h = (
        points_game_official - points_1h
        if pd.notna(points_game_official)
        else points_2h_reg + ot
    )

    segment_identity_ok = (
        pd.notna(points_game_official)
        and abs((points_1h + points_2h_reg + ot) - points_game_official) < 1e-9
    )

    out.update(
        {
            "line_scores_parsed_ok": True,
            "n_periods": float(n),
            "points_1h": float(points_1h),
            "points_2h_reg": float(points_2h_reg),
            "points_ot": float(ot),
            "points_2h": float(points_2h),
            "points_game_linesum": float(points_game_linesum),
            "points_game_official": points_game_official,
            "segment_identity_ok": bool(segment_identity_ok),
            "has_four_quarters": bool(n >= 4),
        }
    )
    return out


def actual_home_cover(home_margin: float, spread: float | None) -> float | None:
    if spread is None or pd.isna(spread):
        return None
    return float(home_margin > (-float(spread)))


def actual_over(total_points: float, total_line: float | None) -> float | None:
    if total_line is None or pd.isna(total_line):
        return None
    return float(total_points > float(total_line))


def main() -> None:
    args = parse_args()

    market_path = Path(args.market_input)
    if not market_path.exists():
        raise FileNotFoundError(f"Market input not found: {market_path}")

    market_df = pd.read_csv(market_path)
    if "game_id" not in market_df.columns:
        raise ValueError("Market input must contain game_id")

    market_df = normalize_game_id(market_df)

    seasons = sorted(pd.to_numeric(market_df["season"], errors="coerce").dropna().astype(int).unique().tolist())
    if not seasons:
        raise ValueError("No seasons found in market input")

    detail_dir = Path(args.detail_dir)
    detail_parts = []

    for season in seasons:
        print(f"Loading detailed games for season {season}...")
        detail_parts.append(fetch_or_load_detailed_games(season, args.season_type, detail_dir))

    detail_df = pd.concat(detail_parts, ignore_index=True)
    detail_df = rename_common_game_cols(detail_df)
    detail_df = normalize_game_id(detail_df)

    keep_cols = [
        c for c in [
            "game_id",
            "season",
            "week",
            "season_type",
            "start_date",
            "neutral_site",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
            "home_line_scores",
            "away_line_scores",
        ]
        if c in detail_df.columns
    ]
    detail_df = detail_df[keep_cols].copy()

    merged = market_df.merge(detail_df, on="game_id", how="left", suffixes=("", "_detail"))

    for col in ["season", "week", "season_type", "start_date", "neutral_site", "home_team", "away_team"]:
        det_col = f"{col}_detail"
        if det_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[det_col])

    if "home_points_detail" in merged.columns:
        merged["home_points_detail"] = pd.to_numeric(merged["home_points_detail"], errors="coerce")
    if "away_points_detail" in merged.columns:
        merged["away_points_detail"] = pd.to_numeric(merged["away_points_detail"], errors="coerce")

    if "home_score" not in merged.columns:
        merged["home_score"] = np.nan
    if "away_score" not in merged.columns:
        merged["away_score"] = np.nan

    merged["home_score"] = pd.to_numeric(merged["home_score"], errors="coerce")
    merged["away_score"] = pd.to_numeric(merged["away_score"], errors="coerce")

    home_segments = []
    away_segments = []

    for _, row in merged.iterrows():
        home_scores = parse_line_scores(row["home_line_scores"]) if "home_line_scores" in row.index else None
        away_scores = parse_line_scores(row["away_line_scores"]) if "away_line_scores" in row.index else None

        home_final = row["home_score"] if pd.notna(row["home_score"]) else row.get("home_points_detail")
        away_final = row["away_score"] if pd.notna(row["away_score"]) else row.get("away_points_detail")

        home_segments.append(extract_segment_points(home_scores, home_final))
        away_segments.append(extract_segment_points(away_scores, away_final))

    home_seg_df = pd.DataFrame(home_segments).add_prefix("home_")
    away_seg_df = pd.DataFrame(away_segments).add_prefix("away_")

    out = pd.concat([merged.reset_index(drop=True), home_seg_df, away_seg_df], axis=1)

    out["home_points_1h"] = out["home_points_1h"]
    out["away_points_1h"] = out["away_points_1h"]

    out["home_points_2h_reg"] = out["home_points_2h_reg"]
    out["away_points_2h_reg"] = out["away_points_2h_reg"]

    out["home_points_ot"] = out["home_points_ot"]
    out["away_points_ot"] = out["away_points_ot"]

    out["home_points_2h"] = out["home_points_2h"]
    out["away_points_2h"] = out["away_points_2h"]

    out["home_points_game"] = out["home_score"]
    out["away_points_game"] = out["away_score"]

    out["margin_1h"] = out["home_points_1h"] - out["away_points_1h"]
    out["margin_2h_reg"] = out["home_points_2h_reg"] - out["away_points_2h_reg"]
    out["margin_ot"] = out["home_points_ot"] - out["away_points_ot"]
    out["margin_2h"] = out["home_points_2h"] - out["away_points_2h"]
    out["margin_game"] = out["home_points_game"] - out["away_points_game"]

    out["line_scores_parsed_ok"] = out["home_line_scores_parsed_ok"] & out["away_line_scores_parsed_ok"]
    out["segment_identity_ok"] = out["home_segment_identity_ok"] & out["away_segment_identity_ok"]
    out["has_four_quarters"] = out["home_has_four_quarters"] & out["away_has_four_quarters"]

    out["score_identity_ok"] = (
        (out["home_points_1h"] + out["home_points_2h"]).round(6) == out["home_points_game"].round(6)
    ) & (
        (out["away_points_1h"] + out["away_points_2h"]).round(6) == out["away_points_game"].round(6)
    )

    out["regulation_identity_ok"] = (
        (out["home_points_1h"] + out["home_points_2h_reg"] + out["home_points_ot"]).round(6) == out["home_points_game"].round(6)
    ) & (
        (out["away_points_1h"] + out["away_points_2h_reg"] + out["away_points_ot"]).round(6) == out["away_points_game"].round(6)
    )

    out["margin_identity_ok"] = (
        (out["margin_1h"] + out["margin_2h"]).round(6) == out["margin_game"].round(6)
    )
    out["regulation_margin_identity_ok"] = (
        (out["margin_1h"] + out["margin_2h_reg"] + out["margin_ot"]).round(6) == out["margin_game"].round(6)
    )

    out["market_spread_game"] = pd.to_numeric(out["market_spread"], errors="coerce") if "market_spread" in out.columns else np.nan
    out["market_total_game"] = pd.to_numeric(out["market_total"], errors="coerce") if "market_total" in out.columns else np.nan

    out["market_spread_1h"] = pd.to_numeric(out["market_spread_1h"], errors="coerce") if "market_spread_1h" in out.columns else np.nan
    out["market_total_1h"] = pd.to_numeric(out["market_total_1h"], errors="coerce") if "market_total_1h" in out.columns else np.nan
    out["market_spread_2h"] = pd.to_numeric(out["market_spread_2h"], errors="coerce") if "market_spread_2h" in out.columns else np.nan
    out["market_total_2h"] = pd.to_numeric(out["market_total_2h"], errors="coerce") if "market_total_2h" in out.columns else np.nan

    out["actual_home_cover_game"] = [
        actual_home_cover(m, s) for m, s in zip(out["margin_game"], out["market_spread_game"])
    ]
    out["actual_over_game"] = [
        actual_over(t, line) for t, line in zip(out["home_points_game"] + out["away_points_game"], out["market_total_game"])
    ]

    out["actual_home_cover_1h"] = [
        actual_home_cover(m, s) for m, s in zip(out["margin_1h"], out["market_spread_1h"])
    ]
    out["actual_over_1h"] = [
        actual_over(t, line) for t, line in zip(out["home_points_1h"] + out["away_points_1h"], out["market_total_1h"])
    ]

    out["actual_home_cover_2h"] = [
        actual_home_cover(m, s) for m, s in zip(out["margin_2h_reg"], out["market_spread_2h"])
    ]
    out["actual_over_2h"] = [
        actual_over(t, line) for t, line in zip(out["home_points_2h_reg"] + out["away_points_2h_reg"], out["market_total_2h"])
    ]

    out["trainable_1h_pmf"] = out["line_scores_parsed_ok"] & out["has_four_quarters"] & out["regulation_identity_ok"]
    out["trainable_2h_pmf"] = out["line_scores_parsed_ok"] & out["has_four_quarters"] & out["regulation_identity_ok"]
    out["trainable_game_pmf"] = out["line_scores_parsed_ok"] & out["score_identity_ok"] & out["margin_identity_ok"]

    final_cols = [
        c for c in [
            "game_id",
            "season",
            "week",
            "season_type",
            "start_date",
            "neutral_site",
            "home_team",
            "away_team",
            "home_points_1h",
            "away_points_1h",
            "home_points_2h_reg",
            "away_points_2h_reg",
            "home_points_ot",
            "away_points_ot",
            "home_points_2h",
            "away_points_2h",
            "home_points_game",
            "away_points_game",
            "margin_1h",
            "margin_2h_reg",
            "margin_ot",
            "margin_2h",
            "margin_game",
            "market_spread_game",
            "market_total_game",
            "market_spread_1h",
            "market_total_1h",
            "market_spread_2h",
            "market_total_2h",
            "actual_home_cover_game",
            "actual_over_game",
            "actual_home_cover_1h",
            "actual_over_1h",
            "actual_home_cover_2h",
            "actual_over_2h",
            "line_scores_parsed_ok",
            "segment_identity_ok",
            "score_identity_ok",
            "regulation_identity_ok",
            "margin_identity_ok",
            "regulation_margin_identity_ok",
            "trainable_1h_pmf",
            "trainable_2h_pmf",
            "trainable_game_pmf",
        ]
        if c in out.columns
    ]

    final_df = out[final_cols].copy()
    final_df = final_df.sort_values(["season", "week", "start_date", "game_id"]).reset_index(drop=True)

    output_csv = Path(args.output_csv)
    output_parquet = Path(args.output_parquet)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(output_csv, index=False)
    final_df.to_parquet(output_parquet, index=False)

    print(f"Wrote {len(final_df)} rows to {output_csv}")
    print(f"Wrote {len(final_df)} rows to {output_parquet}")
    print("\nQA rates:")
    qa_cols = [
        "line_scores_parsed_ok",
        "segment_identity_ok",
        "score_identity_ok",
        "regulation_identity_ok",
        "margin_identity_ok",
        "regulation_margin_identity_ok",
        "trainable_1h_pmf",
        "trainable_2h_pmf",
        "trainable_game_pmf",
    ]
    print(final_df[qa_cols].mean(numeric_only=True))
    print("\nSample:")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
