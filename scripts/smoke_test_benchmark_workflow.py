from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks import (  # noqa: E402
    benchmark_game_probs,
    run_massey_benchmark_for_game,
    run_pi_benchmark_for_game,
)


games_df = pd.DataFrame(
    [
        {"season": 2025, "week": 1, "game_date": "2025-08-30", "home_team": "North Dakota State", "away_team": "Montana", "home_score": 31, "away_score": 21, "neutral_site": False},
        {"season": 2025, "week": 1, "game_date": "2025-08-30", "home_team": "Montana State", "away_team": "South Dakota State", "home_score": 24, "away_score": 27, "neutral_site": False},
        {"season": 2025, "week": 2, "game_date": "2025-09-06", "home_team": "Montana", "away_team": "Montana State", "home_score": 20, "away_score": 28, "neutral_site": False},
        {"season": 2025, "week": 2, "game_date": "2025-09-06", "home_team": "South Dakota State", "away_team": "North Dakota State", "home_score": 17, "away_score": 20, "neutral_site": False},
        {"season": 2025, "week": 3, "game_date": "2025-09-13", "home_team": "North Dakota State", "away_team": "Montana State", "home_score": 27, "away_score": 24, "neutral_site": False},
        {"season": 2025, "week": 3, "game_date": "2025-09-13", "home_team": "South Dakota State", "away_team": "Montana", "home_score": 30, "away_score": 14, "neutral_site": False},
        {"season": 2025, "week": 4, "game_date": "2025-09-20", "home_team": "Montana State", "away_team": "Montana", "home_score": 26, "away_score": 17, "neutral_site": False},
        {"season": 2025, "week": 4, "game_date": "2025-09-20", "home_team": "North Dakota State", "away_team": "South Dakota State", "home_score": 21, "away_score": 23, "neutral_site": False},
    ]
)

home_team = "North Dakota State"
away_team = "Montana State"

massey_result = run_massey_benchmark_for_game(
    games_df=games_df,
    home_team=home_team,
    away_team=away_team,
    ridge=10.0,
    recency_halflife_weeks=2.0,
)

pi_result = run_pi_benchmark_for_game(
    games_df=games_df,
    home_team=home_team,
    away_team=away_team,
    rho_offense=0.90,
    rho_defense=0.90,
    k_offense=0.10,
    k_defense=0.10,
)

massey_probs = benchmark_game_probs(
    massey_result,
    spread=3.5,
    total=52.5,
    home_team_total=27.5,
    away_team_total=24.5,
)

pi_probs = benchmark_game_probs(
    pi_result,
    spread=3.5,
    total=52.5,
    home_team_total=27.5,
    away_team_total=24.5,
)

for label, result, probs in [
    ("Massey", massey_result, massey_probs),
    ("Pi", pi_result, pi_probs),
]:
    print(f"\n=== {label} benchmark ===")
    print(f"Expected score: {result.home_team} {result.expected_home_score:.2f} - {result.away_team} {result.expected_away_score:.2f}")
    print(f"Expected margin (home): {result.expected_margin_home:.2f}")
    print(f"Expected total: {result.expected_total:.2f}")
    print(f"Home ML prob: {probs['home_ml_prob']:.4f}")
    print(f"Away ML prob: {probs['away_ml_prob']:.4f}")
    print(f"Tie prob: {probs['tie_prob']:.4f}")
    print(f"Home -3.5 cover prob: {probs['home_minus_3.5_cover_prob']:.4f}")
    print(f"Away +3.5 cover prob: {probs['away_plus_3.5_cover_prob']:.4f}")
    print(f"Over 52.5 prob: {probs['over_52.5_prob']:.4f}")
    print(f"Under 52.5 prob: {probs['under_52.5_prob']:.4f}")

    assert result.joint_pmf.pmf.ndim == 2
    assert abs(result.joint_pmf.pmf.sum() - 1.0) < 1e-9
    assert 0.0 <= probs["home_ml_prob"] <= 1.0
    assert 0.0 <= probs["away_ml_prob"] <= 1.0
    assert 0.0 <= probs["tie_prob"] <= 1.0
    assert abs(result.expected_total - (result.expected_home_score + result.expected_away_score)) < 1e-9

print("\nSmoke test passed.")
