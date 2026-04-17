from __future__ import annotations

import numpy as np
import pandas as pd


def validate_score_identity(df: pd.DataFrame) -> pd.Series:
    required = [
        "home_points_1h",
        "away_points_1h",
        "home_points_2h",
        "away_points_2h",
        "home_points_game",
        "away_points_game",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required score identity columns: {missing}")

    ok_home = pd.to_numeric(df["home_points_1h"], errors="coerce") + pd.to_numeric(df["home_points_2h"], errors="coerce") == pd.to_numeric(df["home_points_game"], errors="coerce")
    ok_away = pd.to_numeric(df["away_points_1h"], errors="coerce") + pd.to_numeric(df["away_points_2h"], errors="coerce") == pd.to_numeric(df["away_points_game"], errors="coerce")
    return ok_home & ok_away


def validate_margin_identity(df: pd.DataFrame) -> pd.Series:
    required = ["margin_1h", "margin_2h", "margin_game"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required margin identity columns: {missing}")

    lhs = pd.to_numeric(df["margin_1h"], errors="coerce") + pd.to_numeric(df["margin_2h"], errors="coerce")
    rhs = pd.to_numeric(df["margin_game"], errors="coerce")
    return lhs == rhs


def assert_probability_vector(p: np.ndarray, atol: float = 1e-9) -> None:
    if np.any(p < -atol):
        raise ValueError("Probability vector contains negative mass.")
    if not np.isclose(p.sum(), 1.0, atol=atol):
        raise ValueError(f"Probability vector does not sum to 1. Sum={p.sum()}")
