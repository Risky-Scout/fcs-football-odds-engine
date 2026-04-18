from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmf.segment_baseline import (
    combine_segments_to_game_pmf,
    fit_segment_nb_model,
    predict_segment_pmf,
    prob_home_cover_from_margin_pmf,
    prob_over_from_score_pmf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward baseline evaluation for 1H / 2H / game score and margin PMFs.")
    parser.add_argument("--input", default="data/processed/fcs_margin_targets.csv")
    parser.add_argument("--output-prefix", default="outputs/pmf_baseline/fcs_segment_pmf_baseline")
    parser.add_argument("--min-train-games", type=int, default=200)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--support-max", type=int, default=80)
    return parser.parse_args()


def load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    df = pd.read_csv(p)
    sort_cols = [c for c in ["season", "week", "start_date"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def score_pair_logprob(score_pmf, home_points: int, away_points: int, eps: float = 1e-15) -> float:
    h = int(home_points)
    a = int(away_points)
    if h < 0 or a < 0:
        return np.nan
    if h >= len(score_pmf.home_points) or a >= len(score_pmf.away_points):
        return np.log(eps)
    return float(np.log(max(score_pmf.pmf[h, a], eps)))


def margin_logprob(margin_pmf, margin_value: int, eps: float = 1e-15) -> float:
    m = int(margin_value)
    margins = margin_pmf.margins.astype(int)
    idx = np.where(margins == m)[0]
    if len(idx) == 0:
        return np.log(eps)
    return float(np.log(max(margin_pmf.pmf[idx[0]], eps)))


def brier(y_true: pd.Series, p_pred: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(p_pred, errors="coerce")
    mask = y.notna() & p.notna()
    if mask.sum() == 0:
        return float("nan")
    return float(((y[mask] - p[mask]) ** 2).mean())


def summarize(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    segments = [
        ("1h", "actual_home_points_1h", "actual_away_points_1h", "actual_margin_1h", "score_logprob_1h", "margin_logprob_1h"),
        ("2h", "actual_home_points_2h", "actual_away_points_2h", "actual_margin_2h", "score_logprob_2h", "margin_logprob_2h"),
        ("game", "actual_home_points_game", "actual_away_points_game", "actual_margin_game", "score_logprob_game", "margin_logprob_game"),
    ]

    for seg, home_col, away_col, margin_col, score_lp_col, margin_lp_col in segments:
        rows.append(
            {
                "segment": seg,
                "n_rows": int(len(eval_df)),
                "mean_score_logprob": float(eval_df[score_lp_col].mean()),
                "mean_margin_logprob": float(eval_df[margin_lp_col].mean()),
                "home_score_mae": float((eval_df[f"expected_home_{seg}"] - eval_df[home_col]).abs().mean()),
                "away_score_mae": float((eval_df[f"expected_away_{seg}"] - eval_df[away_col]).abs().mean()),
                "margin_mae": float((eval_df[f"expected_margin_{seg}"] - eval_df[margin_col]).abs().mean()),
            }
        )

    market_specs = [
        ("game", "market_spread_game", "actual_home_cover_game", "home_cover_prob_game", "spread_brier"),
        ("game_total", "market_total_game", "actual_over_game", "over_prob_game", "total_brier"),
        ("1h", "market_spread_1h", "actual_home_cover_1h", "home_cover_prob_1h", "spread_brier"),
        ("1h_total", "market_total_1h", "actual_over_1h", "over_prob_1h", "total_brier"),
        ("2h", "market_spread_2h", "actual_home_cover_2h", "home_cover_prob_2h_reg", "spread_brier"),
        ("2h_total", "market_total_2h", "actual_over_2h", "over_prob_2h_reg", "total_brier"),
    ]

    market_rows = []
    for label, line_col, actual_col, prob_col, metric_name in market_specs:
        if line_col not in eval_df.columns or actual_col not in eval_df.columns or prob_col not in eval_df.columns:
            continue
        mask = eval_df[line_col].notna() & eval_df[actual_col].notna() & eval_df[prob_col].notna()
        if mask.sum() == 0:
            continue
        market_rows.append(
            {
                "segment": label,
                "n_rows": int(mask.sum()),
                metric_name: brier(eval_df.loc[mask, actual_col], eval_df.loc[mask, prob_col]),
            }
        )

    score_summary = pd.DataFrame(rows)
    market_summary = pd.DataFrame(market_rows)
    if market_summary.empty:
        return score_summary
    return score_summary.merge(market_summary, on=["segment", "n_rows"], how="left")


def main() -> None:
    args = parse_args()
    df = load_df(args.input)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []

    for i in range(args.min_train_games, len(df)):
        train = df.iloc[:i].copy()
        test = df.iloc[i]

        train = train.loc[
            train["trainable_1h_pmf"].fillna(False)
            & train["trainable_2h_pmf"].fillna(False)
            & train["trainable_game_pmf"].fillna(False)
        ].copy()

        if len(train) < args.min_train_games:
            continue

        known = set(train["home_team"].astype(str)) | set(train["away_team"].astype(str))
        home_team = str(test["home_team"])
        away_team = str(test["away_team"])

        if home_team not in known or away_team not in known:
            skipped.append(
                {
                    "row_number": i,
                    "reason": "unseen_team_in_training",
                    "season": test.get("season"),
                    "week": test.get("week"),
                    "start_date": test.get("start_date"),
                    "home_team": home_team,
                    "away_team": away_team,
                }
            )
            continue

        neutral_site = bool(test["neutral_site"]) if "neutral_site" in test.index and pd.notna(test["neutral_site"]) else False

        model_1h = fit_segment_nb_model(train, "home_points_1h", "away_points_1h", "1h", ridge=args.ridge, support_max=args.support_max)
        model_2h = fit_segment_nb_model(train, "home_points_2h_reg", "away_points_2h_reg", "2h_reg", ridge=args.ridge, support_max=args.support_max)

        pred_1h = predict_segment_pmf(model_1h, home_team, away_team, neutral_site=neutral_site)
        pred_2h = predict_segment_pmf(model_2h, home_team, away_team, neutral_site=neutral_site)
        pred_game = combine_segments_to_game_pmf(pred_1h, pred_2h, segment_name="game")

        actual_home_1h = int(test["home_points_1h"])
        actual_away_1h = int(test["away_points_1h"])
        actual_home_2h = int(test["home_points_2h_reg"])
        actual_away_2h = int(test["away_points_2h_reg"])
        actual_home_game = int(test["home_points_game"])
        actual_away_game = int(test["away_points_game"])

        actual_margin_1h = int(test["margin_1h"])
        actual_margin_2h = int(test["margin_2h_reg"])
        actual_margin_game = int(test["margin_game"])

        row = {
            "season": test.get("season"),
            "week": test.get("week"),
            "start_date": test.get("start_date"),
            "neutral_site": test.get("neutral_site"),
            "home_team": home_team,
            "away_team": away_team,
            "actual_home_points_1h": actual_home_1h,
            "actual_away_points_1h": actual_away_1h,
            "actual_home_points_2h": actual_home_2h,
            "actual_away_points_2h": actual_away_2h,
            "actual_home_points_game": actual_home_game,
            "actual_away_points_game": actual_away_game,
            "actual_margin_1h": actual_margin_1h,
            "actual_margin_2h": actual_margin_2h,
            "actual_margin_game": actual_margin_game,
            "expected_home_1h": pred_1h.expected_home,
            "expected_away_1h": pred_1h.expected_away,
            "expected_home_2h": pred_2h.expected_home,
            "expected_away_2h": pred_2h.expected_away,
            "expected_home_game": pred_game.expected_home,
            "expected_away_game": pred_game.expected_away,
            "expected_margin_1h": pred_1h.expected_home - pred_1h.expected_away,
            "expected_margin_2h": pred_2h.expected_home - pred_2h.expected_away,
            "expected_margin_game": pred_game.expected_home - pred_game.expected_away,
            "score_logprob_1h": score_pair_logprob(pred_1h.score_pmf, actual_home_1h, actual_away_1h),
            "score_logprob_2h": score_pair_logprob(pred_2h.score_pmf, actual_home_2h, actual_away_2h),
            "score_logprob_game": score_pair_logprob(pred_game.score_pmf, actual_home_game, actual_away_game),
            "margin_logprob_1h": margin_logprob(pred_1h.margin_pmf, actual_margin_1h),
            "margin_logprob_2h": margin_logprob(pred_2h.margin_pmf, actual_margin_2h),
            "margin_logprob_game": margin_logprob(pred_game.margin_pmf, actual_margin_game),
            "market_spread_game": test.get("market_spread_game"),
            "market_total_game": test.get("market_total_game"),
            "market_spread_1h": test.get("market_spread_1h"),
            "market_total_1h": test.get("market_total_1h"),
            "market_spread_2h": test.get("market_spread_2h"),
            "market_total_2h": test.get("market_total_2h"),
            "actual_home_cover_game": test.get("actual_home_cover_game"),
            "actual_over_game": test.get("actual_over_game"),
            "actual_home_cover_1h": test.get("actual_home_cover_1h"),
            "actual_over_1h": test.get("actual_over_1h"),
            "actual_home_cover_2h": test.get("actual_home_cover_2h"),
            "actual_over_2h": test.get("actual_over_2h"),
        }

        row["home_cover_prob_game"] = prob_home_cover_from_margin_pmf(pred_game.margin_pmf, float(test["market_spread_game"])) if pd.notna(test.get("market_spread_game")) else np.nan
        row["over_prob_game"] = prob_over_from_score_pmf(pred_game.score_pmf, float(test["market_total_game"])) if pd.notna(test.get("market_total_game")) else np.nan
        row["home_cover_prob_1h"] = prob_home_cover_from_margin_pmf(pred_1h.margin_pmf, float(test["market_spread_1h"])) if pd.notna(test.get("market_spread_1h")) else np.nan
        row["over_prob_1h"] = prob_over_from_score_pmf(pred_1h.score_pmf, float(test["market_total_1h"])) if pd.notna(test.get("market_total_1h")) else np.nan
        row["home_cover_prob_2h_reg"] = prob_home_cover_from_margin_pmf(pred_2h.margin_pmf, float(test["market_spread_2h"])) if pd.notna(test.get("market_spread_2h")) else np.nan
        row["over_prob_2h_reg"] = prob_over_from_score_pmf(pred_2h.score_pmf, float(test["market_total_2h"])) if pd.notna(test.get("market_total_2h")) else np.nan

        rows.append(row)

    eval_df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)
    summary_df = summarize(eval_df) if not eval_df.empty else pd.DataFrame()

    eval_path = output_prefix.with_name(output_prefix.name + "_eval.csv")
    skipped_path = output_prefix.with_name(output_prefix.name + "_skipped.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")

    eval_df.to_csv(eval_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote eval rows to {eval_path}")
    print(f"Wrote skipped rows to {skipped_path}")
    print(f"Wrote summary to {summary_path}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
