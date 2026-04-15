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
class PiRatingsModel:
    teams: list[str]
    team_to_idx: dict[str, int]
    intercept: float
    home_field: float
    offense: np.ndarray
    defense: np.ndarray
    rho_offense: float
    rho_defense: float
    k_offense: float
    k_defense: float
    pregame_predictions: pd.DataFrame

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


def _sort_games_for_sequential_fit(games_df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = []
    for col in ["season", "week", "game_date"]:
        if col in games_df.columns:
            sort_cols.append(col)

    if sort_cols:
        return games_df.sort_values(sort_cols).reset_index(drop=True)

    return games_df.reset_index(drop=True)


def fit_pi_scores(
    games_df: pd.DataFrame,
    rho_offense: float = 0.90,
    rho_defense: float = 0.90,
    k_offense: float = 0.10,
    k_defense: float = 0.10,
    initial_offense: float = 0.0,
    initial_defense: float = 0.0,
    home_field: float | None = None,
    intercept: float | None = None,
) -> PiRatingsModel:
    """
    Fit a simple Pi-style dynamic split offense-defense model.

    Pregame score equations:
        home_score_hat = intercept + home_field + O_home - D_away
        away_score_hat = intercept + O_away - D_home

    Sequential updates:
        O_home <- rho_offense * O_home + k_offense * e_home
        D_away <- rho_defense * D_away - k_defense * e_home
        O_away <- rho_offense * O_away + k_offense * e_away
        D_home <- rho_defense * D_home - k_defense * e_away
    """
    _validate_games_df(games_df)

    df = _sort_games_for_sequential_fit(games_df.copy())
    df["home_score"] = df["home_score"].astype(float)
    df["away_score"] = df["away_score"].astype(float)

    teams, team_to_idx = _team_index(
        pd.concat([df["home_team"], df["away_team"]], ignore_index=True)
    )
    n_teams = len(teams)

    if intercept is None:
        intercept = float(
            pd.concat([df["home_score"], df["away_score"]], ignore_index=True).mean()
        )

    if home_field is None:
        home_field = float((df["home_score"] - df["away_score"]).mean())

    offense = np.full(n_teams, float(initial_offense), dtype=float)
    defense = np.full(n_teams, float(initial_defense), dtype=float)

    prediction_rows: list[dict[str, float | int | str]] = []

    for idx, row in df.iterrows():
        h = team_to_idx[row["home_team"]]
        a = team_to_idx[row["away_team"]]

        neutral_site = bool(row["neutral_site"]) if "neutral_site" in row else False
        hfa = 0.0 if neutral_site else home_field

        old_offense = offense.copy()
        old_defense = defense.copy()

        pred_home = intercept + hfa + old_offense[h] - old_defense[a]
        pred_away = intercept + old_offense[a] - old_defense[h]

        err_home = float(row["home_score"] - pred_home)
        err_away = float(row["away_score"] - pred_away)

        prediction_row: dict[str, float | int | str] = {
            "game_number": idx,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "expected_home_score": float(pred_home),
            "expected_away_score": float(pred_away),
            "expected_margin_home": float(pred_home - pred_away),
            "expected_total": float(pred_home + pred_away),
            "actual_home_score": float(row["home_score"]),
            "actual_away_score": float(row["away_score"]),
            "home_error": err_home,
            "away_error": err_away,
        }

        for col in ["season", "week", "game_date"]:
            if col in row.index:
                prediction_row[col] = row[col]

        prediction_rows.append(prediction_row)

        offense[h] = rho_offense * old_offense[h] + k_offense * err_home
        defense[a] = rho_defense * old_defense[a] - k_defense * err_home

        offense[a] = rho_offense * old_offense[a] + k_offense * err_away
        defense[h] = rho_defense * old_defense[h] - k_defense * err_away

        # enforce identifiability after each update
        offense -= offense.mean()
        defense -= defense.mean()

    pregame_predictions = pd.DataFrame(prediction_rows)

    return PiRatingsModel(
        teams=teams,
        team_to_idx=team_to_idx,
        intercept=float(intercept),
        home_field=float(home_field),
        offense=offense,
        defense=defense,
        rho_offense=float(rho_offense),
        rho_defense=float(rho_defense),
        k_offense=float(k_offense),
        k_defense=float(k_defense),
        pregame_predictions=pregame_predictions,
    )


def predict_schedule(
    model: PiRatingsModel,
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
    "PiRatingsModel",
    "fit_pi_scores",
    "predict_schedule",
]
