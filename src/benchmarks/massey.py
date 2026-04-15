from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_GAME_COLUMNS = {
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}


@dataclass
class MasseyScoreModel:
    teams: list[str]
    team_to_idx: dict[str, int]
    intercept: float
    home_field: float
    offense: np.ndarray
    defense: np.ndarray
    ridge: float
    recency_halflife_weeks: float | None = None

    def predict_expected_scores(
        self,
        home_team: str,
        away_team: str,
        neutral_site: bool = False,
    ) -> tuple[float, float]:
        if home_team not in self.team_to_idx:
            raise KeyError(f"Unknown home team: {home_team}")
        if away_team not in self.team_to_idx:
            raise KeyError(f"Unknown away team: {away_team}")

        h = self.team_to_idx[home_team]
        a = self.team_to_idx[away_team]
        hfa = 0.0 if neutral_site else self.home_field

        mu_home = self.intercept + hfa + self.offense[h] - self.defense[a]
        mu_away = self.intercept + self.offense[a] - self.defense[h]

        return float(mu_home), float(mu_away)

    def predict_game(
        self,
        home_team: str,
        away_team: str,
        neutral_site: bool = False,
    ) -> dict[str, float]:
        mu_home, mu_away = self.predict_expected_scores(
            home_team=home_team,
            away_team=away_team,
            neutral_site=neutral_site,
        )
        return {
            "home_team": home_team,
            "away_team": away_team,
            "expected_home_score": mu_home,
            "expected_away_score": mu_away,
            "expected_margin_home": mu_home - mu_away,
            "expected_total": mu_home + mu_away,
        }

    def ratings_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "team": self.teams,
                "offense_rating": self.offense,
                "defense_rating": self.defense,
            }
        )
        df["net_rating"] = df["offense_rating"] - df["defense_rating"]
        return df.sort_values("net_rating", ascending=False).reset_index(drop=True)


def _validate_games_df(games_df: pd.DataFrame) -> None:
    missing = REQUIRED_GAME_COLUMNS - set(games_df.columns)
    if missing:
        raise ValueError(f"games_df is missing required columns: {sorted(missing)}")

    if games_df.empty:
        raise ValueError("games_df is empty")

    if games_df[["home_team", "away_team"]].isnull().any().any():
        raise ValueError("games_df contains null team names")

    if games_df[["home_score", "away_score"]].isnull().any().any():
        raise ValueError("games_df contains null scores")


def _team_index(teams: Iterable[str]) -> tuple[list[str], dict[str, int]]:
    team_list = sorted(set(teams))
    return team_list, {team: i for i, team in enumerate(team_list)}


def _build_weights(
    games_df: pd.DataFrame,
    recency_halflife_weeks: float | None,
) -> np.ndarray:
    if recency_halflife_weeks is None:
        return np.ones(len(games_df), dtype=float)

    if "week" not in games_df.columns:
        raise ValueError("games_df must contain 'week' when recency_halflife_weeks is used")

    max_week = games_df["week"].max()
    delta = max_week - games_df["week"].to_numpy(dtype=float)
    weights = 0.5 ** (delta / recency_halflife_weeks)
    return weights.astype(float)


def fit_massey_scores(
    games_df: pd.DataFrame,
    ridge: float = 10.0,
    recency_halflife_weeks: float | None = None,
) -> MasseyScoreModel:
    """
    Fit a split offense-defense Massey-style score model.

    Expected score equations:
        home_score = intercept + home_field + O_home - D_away + error
        away_score = intercept + O_away - D_home + error
    """
    _validate_games_df(games_df)

    df = games_df.copy()
    df["home_score"] = df["home_score"].astype(float)
    df["away_score"] = df["away_score"].astype(float)

    teams, team_to_idx = _team_index(
        pd.concat([df["home_team"], df["away_team"]], ignore_index=True)
    )
    n_teams = len(teams)

    n_rows = 2 * len(df)
    n_cols = 2 + 2 * n_teams  # intercept, home_field, offense block, defense block

    X = np.zeros((n_rows, n_cols), dtype=float)
    y = np.zeros(n_rows, dtype=float)

    weights_games = _build_weights(df, recency_halflife_weeks)
    weights_rows = np.repeat(weights_games, 2)

    for g, (_, row) in enumerate(df.iterrows()):
        h = team_to_idx[row["home_team"]]
        a = team_to_idx[row["away_team"]]

        row_home = 2 * g
        row_away = 2 * g + 1

        # intercept
        X[row_home, 0] = 1.0
        X[row_away, 0] = 1.0

        # home field
        X[row_home, 1] = 1.0
        X[row_away, 1] = 0.0

        # offense block
        X[row_home, 2 + h] = 1.0
        X[row_away, 2 + a] = 1.0

        # defense block
        X[row_home, 2 + n_teams + a] = -1.0
        X[row_away, 2 + n_teams + h] = -1.0

        y[row_home] = row["home_score"]
        y[row_away] = row["away_score"]

    W = np.diag(weights_rows)
    xtwx = X.T @ W @ X
    xtwy = X.T @ W @ y

    penalty = np.zeros((n_cols, n_cols), dtype=float)
    for j in range(2, n_cols):
        penalty[j, j] = ridge

    beta = np.linalg.solve(xtwx + penalty, xtwy)

    intercept = float(beta[0])
    home_field = float(beta[1])
    offense = beta[2 : 2 + n_teams].copy()
    defense = beta[2 + n_teams :].copy()

    # enforce identifiability post-fit
    offense -= offense.mean()
    defense -= defense.mean()

    return MasseyScoreModel(
        teams=teams,
        team_to_idx=team_to_idx,
        intercept=intercept,
        home_field=home_field,
        offense=offense,
        defense=defense,
        ridge=ridge,
        recency_halflife_weeks=recency_halflife_weeks,
    )


def predict_schedule(
    model: MasseyScoreModel,
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    required = {"home_team", "away_team"}
    missing = required - set(schedule_df.columns)
    if missing:
        raise ValueError(f"schedule_df is missing required columns: {sorted(missing)}")

    rows: list[dict[str, float | str]] = []
    for _, row in schedule_df.iterrows():
        pred = model.predict_game(
            home_team=row["home_team"],
            away_team=row["away_team"],
            neutral_site=bool(row["neutral_site"]) if "neutral_site" in row else False,
        )
        rows.append(pred)

    return pd.DataFrame(rows)


__all__ = [
    "MasseyScoreModel",
    "fit_massey_scores",
    "predict_schedule",
]
