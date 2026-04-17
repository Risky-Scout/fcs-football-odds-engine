from __future__ import annotations

import pandas as pd


def build_margin_targets(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    required = [
        "home_points_1h",
        "away_points_1h",
        "home_points_game",
        "away_points_game",
    ]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required target columns: {missing}")

    work["home_points_2h"] = pd.to_numeric(work["home_points_game"], errors="coerce") - pd.to_numeric(work["home_points_1h"], errors="coerce")
    work["away_points_2h"] = pd.to_numeric(work["away_points_game"], errors="coerce") - pd.to_numeric(work["away_points_1h"], errors="coerce")

    work["margin_1h"] = pd.to_numeric(work["home_points_1h"], errors="coerce") - pd.to_numeric(work["away_points_1h"], errors="coerce")
    work["margin_2h"] = pd.to_numeric(work["home_points_2h"], errors="coerce") - pd.to_numeric(work["away_points_2h"], errors="coerce")
    work["margin_game"] = pd.to_numeric(work["home_points_game"], errors="coerce") - pd.to_numeric(work["away_points_game"], errors="coerce")

    return work
