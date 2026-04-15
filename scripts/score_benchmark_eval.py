from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "benchmark_name",
    "home_ml_ot_adj_prob",
    "actual_home_win",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score held-out FCS benchmark evaluation output."
    )
    parser.add_argument("--input", required=True, help="Path to benchmark eval CSV.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to summary CSV output.",
    )
    return parser.parse_args()


def load_eval_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    df = pd.read_csv(p)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    return df


def clip_probs(p: pd.Series, eps: float = 1e-12) -> pd.Series:
    return p.clip(lower=eps, upper=1.0 - eps)


def binary_log_loss(y_true: pd.Series, p_pred: pd.Series) -> float:
    p = clip_probs(p_pred)
    y = y_true.astype(float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def brier_score(y_true: pd.Series, p_pred: pd.Series) -> float:
    y = y_true.astype(float)
    p = p_pred.astype(float)
    return float(((p - y) ** 2).mean())


def mean_abs_error(actual: pd.Series, pred: pd.Series) -> float:
    return float((actual.astype(float) - pred.astype(float)).abs().mean())


def score_group(df: pd.DataFrame) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "benchmark_name": str(df["benchmark_name"].iloc[0]),
        "n_games": int(len(df)),
        "moneyline_log_loss": binary_log_loss(df["actual_home_win"], df["home_ml_ot_adj_prob"]),
        "moneyline_brier": brier_score(df["actual_home_win"], df["home_ml_ot_adj_prob"]),
        "mean_expected_home_score": float(df["expected_home_score"].mean()) if "expected_home_score" in df.columns else np.nan,
        "mean_expected_away_score": float(df["expected_away_score"].mean()) if "expected_away_score" in df.columns else np.nan,
        "mean_expected_total": float(df["expected_total"].mean()) if "expected_total" in df.columns else np.nan,
    }

    if {"actual_home_score", "expected_home_score"} <= set(df.columns):
        out["home_score_mae"] = mean_abs_error(df["actual_home_score"], df["expected_home_score"])

    if {"actual_away_score", "expected_away_score"} <= set(df.columns):
        out["away_score_mae"] = mean_abs_error(df["actual_away_score"], df["expected_away_score"])

    if {"actual_home_score", "actual_away_score", "expected_total"} <= set(df.columns):
        actual_total = df["actual_home_score"].astype(float) + df["actual_away_score"].astype(float)
        out["total_mae"] = mean_abs_error(actual_total, df["expected_total"])

    if {"actual_home_cover", "home_spread_cover_prob"} <= set(df.columns):
        spread_df = df.dropna(subset=["actual_home_cover", "home_spread_cover_prob"]).copy()
        out["n_spread_rows"] = int(len(spread_df))
        out["spread_brier"] = (
            brier_score(spread_df["actual_home_cover"], spread_df["home_spread_cover_prob"])
            if len(spread_df) > 0
            else np.nan
        )
    else:
        out["n_spread_rows"] = 0
        out["spread_brier"] = np.nan

    if {"actual_over_total", "over_total_prob"} <= set(df.columns):
        total_df = df.dropna(subset=["actual_over_total", "over_total_prob"]).copy()
        out["n_total_rows"] = int(len(total_df))
        out["total_brier"] = (
            brier_score(total_df["actual_over_total"], total_df["over_total_prob"])
            if len(total_df) > 0
            else np.nan
        )
    else:
        out["n_total_rows"] = 0
        out["total_brier"] = np.nan

    return out


def main() -> None:
    args = parse_args()
    df = load_eval_df(args.input)

    grouped_rows = []
    for benchmark_name, group in df.groupby("benchmark_name", sort=True):
        grouped_rows.append(score_group(group))

    summary_df = pd.DataFrame(grouped_rows).sort_values("benchmark_name").reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    print(f"Wrote benchmark summary to {output_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
