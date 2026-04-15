from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .massey import MasseyScoreModel, fit_massey_scores
from .pi import PiRatingsModel, fit_pi_scores
from .score_pmf_baseline import (
    JointPMF,
    away_margin_greater_than,
    away_win_prob,
    build_independent_joint_pmf,
    fair_american_odds,
    fair_decimal_odds,
    home_margin_greater_than,
    home_win_prob,
    team_total_over_prob,
    team_total_under_prob,
    tie_prob,
    total_over_prob,
    total_under_prob,
)


@dataclass
class BenchmarkGameResult:
    benchmark_name: str
    home_team: str
    away_team: str
    expected_home_score: float
    expected_away_score: float
    expected_margin_home: float
    expected_total: float
    joint_pmf: JointPMF


def _build_game_result(
    benchmark_name: str,
    home_team: str,
    away_team: str,
    mu_home: float,
    mu_away: float,
    kappa_home: float = 18.0,
    kappa_away: float = 18.0,
) -> BenchmarkGameResult:
    joint = build_independent_joint_pmf(
        mu_home=mu_home,
        mu_away=mu_away,
        kappa_home=kappa_home,
        kappa_away=kappa_away,
    )
    return BenchmarkGameResult(
        benchmark_name=benchmark_name,
        home_team=home_team,
        away_team=away_team,
        expected_home_score=joint.expected_home,
        expected_away_score=joint.expected_away,
        expected_margin_home=joint.expected_margin_home,
        expected_total=joint.expected_total,
        joint_pmf=joint,
    )


def benchmark_game_probs(
    result: BenchmarkGameResult,
    spread: float | None = None,
    total: float | None = None,
    home_team_total: float | None = None,
    away_team_total: float | None = None,
) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "benchmark_name": result.benchmark_name,
        "home_team": result.home_team,
        "away_team": result.away_team,
        "expected_home_score": result.expected_home_score,
        "expected_away_score": result.expected_away_score,
        "expected_margin_home": result.expected_margin_home,
        "expected_total": result.expected_total,
    }

    home_ml_prob = home_win_prob(result.joint_pmf)
    away_ml_prob = away_win_prob(result.joint_pmf)
    draw_prob = tie_prob(result.joint_pmf)

    # Benchmark-only college-football OT adjustment:
    # split regulation tie mass evenly until a dedicated OT benchmark module exists.
    home_ml_ot_adj_prob = home_ml_prob + 0.5 * draw_prob
    away_ml_ot_adj_prob = away_ml_prob + 0.5 * draw_prob

    out["home_ml_prob"] = home_ml_prob
    out["away_ml_prob"] = away_ml_prob
    out["tie_prob"] = draw_prob

    out["home_ml_ot_adj_prob"] = home_ml_ot_adj_prob
    out["away_ml_ot_adj_prob"] = away_ml_ot_adj_prob

    out["home_ml_decimal"] = fair_decimal_odds(home_ml_ot_adj_prob) if 0 < home_ml_ot_adj_prob < 1 else None
    out["away_ml_decimal"] = fair_decimal_odds(away_ml_ot_adj_prob) if 0 < away_ml_ot_adj_prob < 1 else None
    out["home_ml_american"] = fair_american_odds(home_ml_ot_adj_prob) if 0 < home_ml_ot_adj_prob < 1 else None
    out["away_ml_american"] = fair_american_odds(away_ml_ot_adj_prob) if 0 < away_ml_ot_adj_prob < 1 else None

    if spread is not None:
        home_cover = home_margin_greater_than(result.joint_pmf, spread)
        away_cover = away_margin_greater_than(result.joint_pmf, -spread)
        out[f"home_minus_{spread}_cover_prob"] = home_cover
        out[f"away_plus_{spread}_cover_prob"] = away_cover

    if total is not None:
        out[f"over_{total}_prob"] = total_over_prob(result.joint_pmf, total)
        out[f"under_{total}_prob"] = total_under_prob(result.joint_pmf, total)

    if home_team_total is not None:
        out[f"home_team_total_over_{home_team_total}_prob"] = team_total_over_prob(
            result.joint_pmf, "home", home_team_total
        )
        out[f"home_team_total_under_{home_team_total}_prob"] = team_total_under_prob(
            result.joint_pmf, "home", home_team_total
        )

    if away_team_total is not None:
        out[f"away_team_total_over_{away_team_total}_prob"] = team_total_over_prob(
            result.joint_pmf, "away", away_team_total
        )
        out[f"away_team_total_under_{away_team_total}_prob"] = team_total_under_prob(
            result.joint_pmf, "away", away_team_total
        )

    return out


def run_massey_benchmark_for_game(
    games_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    neutral_site: bool = False,
    ridge: float = 10.0,
    recency_halflife_weeks: float | None = None,
    kappa_home: float = 18.0,
    kappa_away: float = 18.0,
) -> BenchmarkGameResult:
    model: MasseyScoreModel = fit_massey_scores(
        games_df=games_df,
        ridge=ridge,
        recency_halflife_weeks=recency_halflife_weeks,
    )
    mu_home, mu_away = model.predict_expected_scores(
        home_team=home_team,
        away_team=away_team,
        neutral_site=neutral_site,
    )
    return _build_game_result(
        benchmark_name="massey",
        home_team=home_team,
        away_team=away_team,
        mu_home=mu_home,
        mu_away=mu_away,
        kappa_home=kappa_home,
        kappa_away=kappa_away,
    )


def run_pi_benchmark_for_game(
    games_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    neutral_site: bool = False,
    rho_offense: float = 0.90,
    rho_defense: float = 0.90,
    k_offense: float = 0.10,
    k_defense: float = 0.10,
    kappa_home: float = 18.0,
    kappa_away: float = 18.0,
) -> BenchmarkGameResult:
    model: PiRatingsModel = fit_pi_scores(
        games_df=games_df,
        rho_offense=rho_offense,
        rho_defense=rho_defense,
        k_offense=k_offense,
        k_defense=k_defense,
    )
    mu_home, mu_away = model.predict_expected_scores(
        home_team=home_team,
        away_team=away_team,
        neutral_site=neutral_site,
    )
    return _build_game_result(
        benchmark_name="pi",
        home_team=home_team,
        away_team=away_team,
        mu_home=mu_home,
        mu_away=mu_away,
        kappa_home=kappa_home,
        kappa_away=kappa_away,
    )


__all__ = [
    "BenchmarkGameResult",
    "benchmark_game_probs",
    "run_massey_benchmark_for_game",
    "run_pi_benchmark_for_game",
]
