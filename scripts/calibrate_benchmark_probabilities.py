from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"benchmark_name"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling out-of-sample calibration for FCS benchmark probabilities."
    )
    parser.add_argument("--input", required=True, help="Path to benchmark eval CSV.")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for output files, e.g. outputs/benchmark_eval/fcs_historical_benchmark_calibrated",
    )
    parser.add_argument(
        "--min-cal-train",
        type=int,
        default=200,
        help="Minimum prior rows required to fit a calibrator for a given benchmark/market.",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of equal-frequency bins for ECE computation.",
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


def sort_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["season", "week", "start_date", "game_date"] if c in df.columns]
    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)
    return df.reset_index(drop=True)


def clip_probs(p: np.ndarray | pd.Series, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    return np.clip(arr, eps, 1.0 - eps)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip_probs(p)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt_scaler(
    probs: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    l2: float = 1e-4,
) -> tuple[float, float]:
    """
    Fit logistic calibration:
        calibrated_prob = sigmoid(a + b * logit(raw_prob))
    using Newton-Raphson with small L2 regularization.
    """
    x = logit(probs)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0], dtype=float)

    for _ in range(max_iter):
        z = X @ beta
        p = sigmoid(z)

        grad = X.T @ (p - y) + l2 * beta
        w = p * (1.0 - p)
        H = X.T @ (X * w[:, None]) + l2 * np.eye(X.shape[1])

        step = np.linalg.solve(H, grad)
        beta_new = beta - step

        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    return float(beta[0]), float(beta[1])


def apply_platt(probs: np.ndarray, a: float, b: float) -> np.ndarray:
    return sigmoid(a + b * logit(probs))


def rolling_calibrate_series(
    probs: pd.Series,
    actuals: pd.Series,
    min_cal_train: int,
) -> pd.Series:
    probs_np = pd.to_numeric(probs, errors="coerce").to_numpy(dtype=float)
    y_np = pd.to_numeric(actuals, errors="coerce").to_numpy(dtype=float)

    out = np.full(len(probs_np), np.nan, dtype=float)

    for i in range(len(probs_np)):
        if np.isnan(probs_np[i]) or np.isnan(y_np[i]):
            continue

        hist_mask = (~np.isnan(probs_np[:i])) & (~np.isnan(y_np[:i]))
        if hist_mask.sum() < min_cal_train:
            continue

        a, b = fit_platt_scaler(probs_np[:i][hist_mask], y_np[:i][hist_mask])
        out[i] = float(apply_platt(np.array([probs_np[i]]), a, b)[0])

    return pd.Series(out, index=probs.index)


def binary_log_loss(y_true: pd.Series, p_pred: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce").astype(float)
    p = clip_probs(pd.to_numeric(p_pred, errors="coerce").astype(float))
    mask = (~np.isnan(y)) & (~np.isnan(p))
    if mask.sum() == 0:
        return float("nan")
    y = y[mask]
    p = p[mask]
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def brier_score(y_true: pd.Series, p_pred: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce").astype(float)
    p = pd.to_numeric(p_pred, errors="coerce").astype(float)
    mask = (~np.isnan(y)) & (~np.isnan(p))
    if mask.sum() == 0:
        return float("nan")
    y = y[mask]
    p = p[mask]
    return float(((p - y) ** 2).mean())


def ece(y_true: pd.Series, p_pred: pd.Series, n_bins: int) -> float:
    work = pd.DataFrame({
        "y": pd.to_numeric(y_true, errors="coerce"),
        "p": pd.to_numeric(p_pred, errors="coerce"),
    }).dropna()

    if work.empty:
        return float("nan")

    work["p"] = clip_probs(work["p"].to_numpy())
    work["bin"] = pd.qcut(work["p"], q=n_bins, labels=False, duplicates="drop")

    grouped = (
        work.groupby("bin", dropna=False)
        .agg(count=("y", "size"), mean_pred=("p", "mean"), empirical_rate=("y", "mean"))
        .reset_index()
    )
    grouped["abs_gap"] = (grouped["mean_pred"] - grouped["empirical_rate"]).abs()
    weights = grouped["count"] / grouped["count"].sum()
    return float((weights * grouped["abs_gap"]).sum())


def summarize_market(
    benchmark_name: str,
    market_type: str,
    actual_col: str,
    raw_col: str,
    cal_col: str,
    df: pd.DataFrame,
    ece_bins: int,
) -> dict[str, float | int | str]:
    sub = df[[actual_col, raw_col, cal_col]].dropna().copy()

    return {
        "benchmark_name": benchmark_name,
        "market_type": market_type,
        "n_rows": int(len(sub)),
        "raw_log_loss": binary_log_loss(sub[actual_col], sub[raw_col]),
        "calibrated_log_loss": binary_log_loss(sub[actual_col], sub[cal_col]),
        "raw_brier": brier_score(sub[actual_col], sub[raw_col]),
        "calibrated_brier": brier_score(sub[actual_col], sub[cal_col]),
        "raw_ece": ece(sub[actual_col], sub[raw_col], ece_bins),
        "calibrated_ece": ece(sub[actual_col], sub[cal_col], ece_bins),
        "raw_mean_pred": float(sub[raw_col].mean()) if len(sub) else np.nan,
        "cal_mean_pred": float(sub[cal_col].mean()) if len(sub) else np.nan,
        "mean_actual": float(sub[actual_col].mean()) if len(sub) else np.nan,
    }


def main() -> None:
    args = parse_args()
    df = sort_eval_df(load_eval_df(args.input)).copy()

    market_specs = [
        ("moneyline", "home_ml_ot_adj_prob", "actual_home_win", "home_ml_ot_adj_prob_cal"),
        ("spread", "home_spread_cover_prob", "actual_home_cover", "home_spread_cover_prob_cal"),
        ("total", "over_total_prob", "actual_over_total", "over_total_prob_cal"),
    ]

    out_parts = []
    summary_rows = []

    for benchmark_name, group in df.groupby("benchmark_name", sort=True):
        g = group.copy().reset_index(drop=False).rename(columns={"index": "_orig_index"})
        g = sort_eval_df(g)

        for market_type, raw_col, actual_col, cal_col in market_specs:
            if raw_col not in g.columns or actual_col not in g.columns:
                continue

            g[cal_col] = rolling_calibrate_series(
                probs=g[raw_col],
                actuals=g[actual_col],
                min_cal_train=args.min_cal_train,
            )

            summary_rows.append(
                summarize_market(
                    benchmark_name=benchmark_name,
                    market_type=market_type,
                    actual_col=actual_col,
                    raw_col=raw_col,
                    cal_col=cal_col,
                    df=g,
                    ece_bins=args.ece_bins,
                )
            )

        out_parts.append(g)

    calibrated = pd.concat(out_parts, ignore_index=True)
    calibrated = calibrated.sort_values("_orig_index").drop(columns=["_orig_index"]).reset_index(drop=True)

    summary = pd.DataFrame(summary_rows).sort_values(["benchmark_name", "market_type"]).reset_index(drop=True)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    calibrated_path = output_prefix.with_name(output_prefix.name + "_eval.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")

    calibrated.to_csv(calibrated_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote calibrated eval to {calibrated_path}")
    print(f"Wrote calibration summary to {summary_path}")
    print("\nCalibration comparison summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
