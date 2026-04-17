from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MARKET_SPECS = {
    "moneyline": {
        "raw_col": "home_ml_ot_adj_prob",
        "cal_col": "home_ml_ot_adj_prob_cal",
        "ref_col": "home_ml_ref_prob",
    },
    "spread": {
        "raw_col": "home_spread_cover_prob",
        "cal_col": "home_spread_cover_prob_cal",
        "ref_col": "home_spread_ref_prob",
    },
    "total": {
        "raw_col": "over_total_prob",
        "cal_col": "over_total_prob_cal",
        "ref_col": "over_total_ref_prob",
    },
}

SUMMARY_REQUIRED = {
    "benchmark_name",
    "market_type",
    "raw_log_loss",
    "calibrated_log_loss",
    "raw_brier",
    "calibrated_brier",
    "raw_ece",
    "calibrated_ece",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best benchmark/source combination per market and build a reference probability file."
    )
    parser.add_argument("--eval-input", required=True, help="Path to calibrated benchmark eval CSV.")
    parser.add_argument("--summary-input", required=True, help="Path to calibrated benchmark summary CSV.")
    parser.add_argument("--output-prefix", required=True, help="Prefix for output files.")
    return parser.parse_args()


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing input file: {p}")
    return pd.read_csv(p)


def choose_variant(row: pd.Series) -> tuple[str, float, float, float]:
    raw = ("raw", float(row["raw_log_loss"]), float(row["raw_brier"]), float(row["raw_ece"]))

    cal_ok = (
        pd.notna(row["calibrated_log_loss"])
        and pd.notna(row["calibrated_brier"])
        and pd.notna(row["calibrated_ece"])
        and float(row["calibrated_log_loss"]) <= float(row["raw_log_loss"])
        and float(row["calibrated_brier"]) <= float(row["raw_brier"])
    )

    if not cal_ok:
        return raw

    cal = (
        "calibrated",
        float(row["calibrated_log_loss"]),
        float(row["calibrated_brier"]),
        float(row["calibrated_ece"]),
    )
    return min([raw, cal], key=lambda x: (x[1], x[2], x[3]))


def build_selection(summary_df: pd.DataFrame) -> pd.DataFrame:
    missing = SUMMARY_REQUIRED - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary input missing columns: {sorted(missing)}")

    rows = []
    for market_type, group in summary_df.groupby("market_type", sort=True):
        candidates = []
        for _, row in group.iterrows():
            source, score_log_loss, score_brier, score_ece = choose_variant(row)
            candidates.append(
                {
                    "market_type": market_type,
                    "benchmark_name": row["benchmark_name"],
                    "source_type": source,
                    "score_log_loss": score_log_loss,
                    "score_brier": score_brier,
                    "score_ece": score_ece,
                    "raw_log_loss": row["raw_log_loss"],
                    "calibrated_log_loss": row["calibrated_log_loss"],
                    "raw_brier": row["raw_brier"],
                    "calibrated_brier": row["calibrated_brier"],
                    "raw_ece": row["raw_ece"],
                    "calibrated_ece": row["calibrated_ece"],
                }
            )

        cand_df = pd.DataFrame(candidates).sort_values(
            ["score_log_loss", "score_brier", "score_ece", "benchmark_name"]
        )
        winner = cand_df.iloc[0].to_dict()
        rows.append(winner)

    return pd.DataFrame(rows).sort_values("market_type").reset_index(drop=True)


def merge_reference_probs(eval_df: pd.DataFrame, selection_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        c for c in [
            "season",
            "week",
            "start_date",
            "home_team",
            "away_team",
            "neutral_site",
            "actual_home_score",
            "actual_away_score",
            "actual_home_win",
            "market_spread",
            "market_total",
            "actual_home_cover",
            "actual_over_total",
        ]
        if c in eval_df.columns
    ]

    base = eval_df[key_cols].drop_duplicates().reset_index(drop=True)

    for _, sel in selection_df.iterrows():
        market_type = sel["market_type"]
        benchmark_name = sel["benchmark_name"]
        source_type = sel["source_type"]

        raw_col = MARKET_SPECS[market_type]["raw_col"]
        cal_col = MARKET_SPECS[market_type]["cal_col"]
        ref_col = MARKET_SPECS[market_type]["ref_col"]

        subset = eval_df.loc[eval_df["benchmark_name"] == benchmark_name, key_cols + [raw_col] + ([cal_col] if cal_col in eval_df.columns else [])].copy()

        if source_type == "calibrated" and cal_col in subset.columns:
            subset[ref_col] = pd.to_numeric(subset[cal_col], errors="coerce").fillna(
                pd.to_numeric(subset[raw_col], errors="coerce")
            )
        else:
            subset[ref_col] = pd.to_numeric(subset[raw_col], errors="coerce")

        subset = subset[key_cols + [ref_col]].drop_duplicates()

        base = base.merge(subset, on=key_cols, how="left")
        base[f"{market_type}_ref_benchmark"] = benchmark_name
        base[f"{market_type}_ref_source"] = source_type

    if "home_spread_ref_prob" in base.columns:
        base["away_spread_ref_prob"] = 1.0 - base["home_spread_ref_prob"]
    if "over_total_ref_prob" in base.columns:
        base["under_total_ref_prob"] = 1.0 - base["over_total_ref_prob"]
    if "home_ml_ref_prob" in base.columns:
        base["away_ml_ref_prob"] = 1.0 - base["home_ml_ref_prob"]

    return base


def main() -> None:
    args = parse_args()

    eval_df = load_csv(args.eval_input)
    summary_df = load_csv(args.summary_input)

    selection_df = build_selection(summary_df)
    reference_df = merge_reference_probs(eval_df, selection_df)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    selection_path = output_prefix.with_name(output_prefix.name + "_selection.csv")
    reference_path = output_prefix.with_name(output_prefix.name + "_eval.csv")

    selection_df.to_csv(selection_path, index=False)
    reference_df.to_csv(reference_path, index=False)

    print(f"Wrote selection summary to {selection_path}")
    print(f"Wrote reference eval to {reference_path}")
    print("\nChosen reference model components:")
    print(selection_df.to_string(index=False))


if __name__ == "__main__":
    main()
