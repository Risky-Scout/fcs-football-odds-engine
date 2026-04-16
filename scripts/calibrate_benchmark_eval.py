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
        description="Create calibration summaries for FCS benchmark evaluation output."
    )
    parser.add_argument("--input", required=True, help="Path to benchmark eval CSV.")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for calibration output files, e.g. outputs/benchmark_eval/fcs_benchmark_calibration",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of equal-frequency bins for calibration tables.",
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


def calibration_bins(
    df: pd.DataFrame,
    prob_col: str,
    actual_col: str,
    n_bins: int,
) -> pd.DataFrame:
    work = df[[prob_col, actual_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "bin",
                "count",
                "mean_pred",
                "empirical_rate",
                "abs_gap",
            ]
        )

    work[prob_col] = clip_probs(work[prob_col])

    # equal-frequency bins; drop duplicates when probabilities are tied heavily
    work["bin"] = pd.qcut(work[prob_col], q=n_bins, labels=False, duplicates="drop")

    out = (
        work.groupby("bin", dropna=False)
        .agg(
            count=(actual_col, "size"),
            mean_pred=(prob_col, "mean"),
            empirical_rate=(actual_col, "mean"),
        )
        .reset_index()
    )
    out["abs_gap"] = (out["mean_pred"] - out["empirical_rate"]).abs()
    return out


def expected_calibration_error(calib_df: pd.DataFrame) -> float:
    if calib_df.empty:
        return float("nan")

    total = calib_df["count"].sum()
    if total <= 0:
        return float("nan")

    weights = calib_df["count"] / total
    return float((weights * calib_df["abs_gap"]).sum())


def summarize_benchmark(df: pd.DataFrame, n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = str(df["benchmark_name"].iloc[0])

    summaries = []
    cal_tables = []

    configs = [
        ("moneyline", "home_ml_ot_adj_prob", "actual_home_win"),
        ("spread", "home_spread_cover_prob", "actual_home_cover"),
        ("total", "over_total_prob", "actual_over_total"),
    ]

    for market_type, prob_col, actual_col in configs:
        if prob_col not in df.columns or actual_col not in df.columns:
            continue

        calib = calibration_bins(df, prob_col, actual_col, n_bins=n_bins)
        calib.insert(0, "benchmark_name", benchmark)
        calib.insert(1, "market_type", market_type)
        cal_tables.append(calib)

        row = {
            "benchmark_name": benchmark,
            "market_type": market_type,
            "n_rows": int(df[[prob_col, actual_col]].dropna().shape[0]),
            "ece": expected_calibration_error(calib),
            "mean_pred": float(df[prob_col].dropna().mean()) if df[prob_col].notna().any() else np.nan,
            "mean_actual": float(df[actual_col].dropna().mean()) if df[actual_col].notna().any() else np.nan,
        }
        summaries.append(row)

    summary_df = pd.DataFrame(summaries)
    cal_df = pd.concat(cal_tables, ignore_index=True) if cal_tables else pd.DataFrame()
    return summary_df, cal_df


def main() -> None:
    args = parse_args()
    df = load_eval_df(args.input)

    summary_parts = []
    calib_parts = []

    for benchmark_name, group in df.groupby("benchmark_name", sort=True):
        summary_df, cal_df = summarize_benchmark(group, n_bins=args.n_bins)
        summary_parts.append(summary_df)
        calib_parts.append(cal_df)

    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    calibration = pd.concat(calib_parts, ignore_index=True) if calib_parts else pd.DataFrame()

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    calibration_path = output_prefix.with_name(output_prefix.name + "_bins.csv")

    summary.to_csv(summary_path, index=False)
    calibration.to_csv(calibration_path, index=False)

    print(f"Wrote calibration summary to {summary_path}")
    print(f"Wrote calibration bins to {calibration_path}")

    if not summary.empty:
        print("\nCalibration summary:")
        print(summary.to_string(index=False))

    if not calibration.empty:
        print("\nCalibration bins sample:")
        print(calibration.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
