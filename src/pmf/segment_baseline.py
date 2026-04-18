from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .types import MarginPMF, ScorePMF


@dataclass
class SegmentNBModel:
    segment_name: str
    teams: list[str]
    team_to_idx: dict[str, int]
    intercept: float
    home_field: float
    offense: np.ndarray
    defense: np.ndarray
    ridge: float
    kappa_home: float
    kappa_away: float
    support_max: int

    def predict_means(
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

        eta_home = self.intercept + hfa + self.offense[h] - self.defense[a]
        eta_away = self.intercept + self.offense[a] - self.defense[h]

        mu_home = max(float(np.exp(eta_home) - 0.5), 0.01)
        mu_away = max(float(np.exp(eta_away) - 0.5), 0.01)
        return mu_home, mu_away


@dataclass
class SegmentPMFResult:
    segment_name: str
    expected_home: float
    expected_away: float
    score_pmf: ScorePMF
    margin_pmf: MarginPMF


def _validate_training_df(df: pd.DataFrame, home_col: str, away_col: str) -> None:
    required = {"home_team", "away_team", home_col, away_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"training df missing required columns: {sorted(missing)}")


def _estimate_kappa(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return 20.0
    mean = float(vals.mean())
    var = float(vals.var(ddof=1)) if len(vals) > 1 else mean
    if mean <= 0:
        return 20.0
    if var <= mean:
        return 500.0
    kappa = mean * mean / max(var - mean, 1e-9)
    return float(np.clip(kappa, 0.5, 500.0))


def _nb_pmf_grid(mu: float, kappa: float, max_points: int) -> np.ndarray:
    if max_points < 0:
        raise ValueError("max_points must be nonnegative")
    if mu <= 0:
        out = np.zeros(max_points + 1, dtype=float)
        out[0] = 1.0
        return out

    r = float(kappa)
    p = r / (r + mu)

    xs = np.arange(max_points + 1, dtype=float)
    from math import exp, lgamma

    probs = np.zeros(max_points + 1, dtype=float)
    for i, x in enumerate(xs):
        log_pmf = (
            lgamma(x + r)
            - lgamma(r)
            - lgamma(x + 1.0)
            + r * np.log(p)
            + x * np.log(1.0 - p)
        )
        probs[i] = exp(log_pmf)

    s = probs.sum()
    if s <= 0:
        probs[:] = 0.0
        probs[0] = 1.0
        return probs
    return probs / s


def _score_pmf_to_margin_pmf(score_pmf: ScorePMF, segment_name: str) -> MarginPMF:
    home_points = score_pmf.home_points
    away_points = score_pmf.away_points
    pmf = score_pmf.pmf

    min_margin = int(home_points.min() - away_points.max())
    max_margin = int(home_points.max() - away_points.min())
    margins = np.arange(min_margin, max_margin + 1, dtype=int)
    margin_p = np.zeros(len(margins), dtype=float)

    for i, h in enumerate(home_points):
        for j, a in enumerate(away_points):
            margin_p[int(h - a) - min_margin] += pmf[i, j]

    s = margin_p.sum()
    if s > 0:
        margin_p /= s

    return MarginPMF(
        margins=margins.astype(float),
        pmf=margin_p,
        segment_name=segment_name,
    )


def fit_segment_nb_model(
    df: pd.DataFrame,
    home_col: str,
    away_col: str,
    segment_name: str,
    ridge: float = 10.0,
    support_max: int | None = None,
) -> SegmentNBModel:
    _validate_training_df(df, home_col, away_col)

    work = df[["home_team", "away_team", home_col, away_col] + [c for c in ["neutral_site"] if c in df.columns]].copy()
    work = work.dropna(subset=["home_team", "away_team", home_col, away_col]).reset_index(drop=True)

    teams = sorted(set(work["home_team"].astype(str)) | set(work["away_team"].astype(str)))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    n_games = len(work)
    n_rows = 2 * n_games
    n_cols = 2 + 2 * n_teams

    X = np.zeros((n_rows, n_cols), dtype=float)
    y = np.zeros(n_rows, dtype=float)

    for g, (_, row) in enumerate(work.iterrows()):
        h = team_to_idx[str(row["home_team"])]
        a = team_to_idx[str(row["away_team"])]

        row_home = 2 * g
        row_away = 2 * g + 1

        neutral = bool(row["neutral_site"]) if "neutral_site" in row.index and pd.notna(row["neutral_site"]) else False
        hfa = 0.0 if neutral else 1.0

        X[row_home, 0] = 1.0
        X[row_home, 1] = hfa
        X[row_home, 2 + h] = 1.0
        X[row_home, 2 + n_teams + a] = -1.0
        y[row_home] = np.log(float(row[home_col]) + 0.5)

        X[row_away, 0] = 1.0
        X[row_away, 1] = 0.0
        X[row_away, 2 + a] = 1.0
        X[row_away, 2 + n_teams + h] = -1.0
        y[row_away] = np.log(float(row[away_col]) + 0.5)

    xtx = X.T @ X
    xty = X.T @ y

    penalty = np.zeros((n_cols, n_cols), dtype=float)
    for j in range(2, n_cols):
        penalty[j, j] = ridge

    beta = np.linalg.solve(xtx + penalty, xty)

    intercept = float(beta[0])
    home_field = float(beta[1])
    offense = beta[2 : 2 + n_teams].copy()
    defense = beta[2 + n_teams :].copy()

    offense -= offense.mean()
    defense -= defense.mean()

    kappa_home = _estimate_kappa(pd.to_numeric(work[home_col], errors="coerce").to_numpy())
    kappa_away = _estimate_kappa(pd.to_numeric(work[away_col], errors="coerce").to_numpy())

    if support_max is None:
        observed_max = max(
            int(pd.to_numeric(work[home_col], errors="coerce").max()),
            int(pd.to_numeric(work[away_col], errors="coerce").max()),
        )
        support_max = max(40, observed_max + 20)

    return SegmentNBModel(
        segment_name=segment_name,
        teams=teams,
        team_to_idx=team_to_idx,
        intercept=intercept,
        home_field=home_field,
        offense=offense,
        defense=defense,
        ridge=ridge,
        kappa_home=kappa_home,
        kappa_away=kappa_away,
        support_max=int(support_max),
    )


def predict_segment_pmf(
    model: SegmentNBModel,
    home_team: str,
    away_team: str,
    neutral_site: bool = False,
) -> SegmentPMFResult:
    mu_home, mu_away = model.predict_means(home_team, away_team, neutral_site=neutral_site)

    home_grid = np.arange(model.support_max + 1, dtype=float)
    away_grid = np.arange(model.support_max + 1, dtype=float)

    home_p = _nb_pmf_grid(mu_home, model.kappa_home, model.support_max)
    away_p = _nb_pmf_grid(mu_away, model.kappa_away, model.support_max)

    joint = np.outer(home_p, away_p)
    joint /= joint.sum()

    score_pmf = ScorePMF(
        home_points=home_grid,
        away_points=away_grid,
        pmf=joint,
        segment_name=model.segment_name,
    )
    margin_pmf = _score_pmf_to_margin_pmf(score_pmf, segment_name=model.segment_name)

    return SegmentPMFResult(
        segment_name=model.segment_name,
        expected_home=mu_home,
        expected_away=mu_away,
        score_pmf=score_pmf,
        margin_pmf=margin_pmf,
    )


def combine_segments_to_game_pmf(
    first_half: SegmentPMFResult,
    second_half: SegmentPMFResult,
    segment_name: str = "game",
) -> SegmentPMFResult:
    home_1h = first_half.score_pmf.pmf.sum(axis=1)
    away_1h = first_half.score_pmf.pmf.sum(axis=0)
    home_2h = second_half.score_pmf.pmf.sum(axis=1)
    away_2h = second_half.score_pmf.pmf.sum(axis=0)

    home_game = np.convolve(home_1h, home_2h)
    away_game = np.convolve(away_1h, away_2h)

    home_points = np.arange(len(home_game), dtype=float)
    away_points = np.arange(len(away_game), dtype=float)

    joint = np.outer(home_game, away_game)
    joint /= joint.sum()

    score_pmf = ScorePMF(
        home_points=home_points,
        away_points=away_points,
        pmf=joint,
        segment_name=segment_name,
    )
    margin_pmf = _score_pmf_to_margin_pmf(score_pmf, segment_name=segment_name)

    return SegmentPMFResult(
        segment_name=segment_name,
        expected_home=float((home_points * home_game).sum()),
        expected_away=float((away_points * away_game).sum()),
        score_pmf=score_pmf,
        margin_pmf=margin_pmf,
    )


def prob_home_cover_from_margin_pmf(margin_pmf: MarginPMF, home_spread: float) -> float:
    threshold = -float(home_spread)
    margins = margin_pmf.margins
    return float(margin_pmf.pmf[margins > threshold].sum())


def prob_over_from_score_pmf(score_pmf: ScorePMF, total_line: float) -> float:
    home = score_pmf.home_points[:, None]
    away = score_pmf.away_points[None, :]
    total = home + away
    return float(score_pmf.pmf[total > float(total_line)].sum())
