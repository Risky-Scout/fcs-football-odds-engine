from __future__ import annotations

import argparse
from pathlib import Path
import sys
from dataclasses import dataclass
from math import exp, lgamma

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.benchmark_workflow import run_pi_benchmark_for_game  # noqa: E402


@dataclass
class JointScorePMF:
    home_points: np.ndarray
    away_points: np.ndarray
    pmf: np.ndarray
    expected_home: float
    expected_away: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward hybrid Pi-anchored evaluation for 1H / 2H / game score and margin PMFs."
    )
    parser.add_argument("--input", default="data/processed/fcs_margin_targets.csv")
    parser.add_argument("--output-prefix", default="outputs/pmf_baseline/fcs_segment_pmf_baseline")
    parser.add_argument("--min-train-games", type=int, default=200)
    parser.add_argument("--support-max", type=int, default=80)
    parser.add_argument("--share-prior-points", type=float, default=150.0)
    parser.add_argument("--share-min", type=float, default=0.30)
    parser.add_argument("--share-max", type=float, default=0.70)
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


def estimate_kappa(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
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


def nb_pmf_grid(mu: float, kappa: float, max_points: int) -> np.ndarray:
    if mu <= 0:
        out = np.zeros(max_points + 1, dtype=float)
        out[0] = 1.0
        return out

    r = float(kappa)
    p = r / (r + mu)

    probs = np.zeros(max_points + 1, dtype=float)
    for x in range(max_points + 1):
        log_pmf = (
            lgamma(x + r)
            - lgamma(r)
            - lgamma(x + 1.0)
            + r * np.log(p)
            + x * np.log(1.0 - p)
        )
        probs[x] = exp(log_pmf)

    s = probs.sum()
    if s <= 0:
        probs[:] = 0.0
        probs[0] = 1.0
        return probs
    return probs / s


def build_joint_pmf(mu_home: float, mu_away: float, kappa_home: float, kappa_away: float, max_points: int) -> JointScorePMF:
    home = nb_pmf_grid(mu_home, kappa_home, max_points)
    away = nb_pmf_grid(mu_away, kappa_away, max_points)
    joint = np.outer(home, away)
    joint /= joint.sum()
    home_points = np.arange(max_points + 1, dtype=float)
    away_points = np.arange(max_points + 1, dtype=float)
    return JointScorePMF(
        home_points=home_points,
        away_points=away_points,
        pmf=joint,
        expected_home=float((home_points * home).sum()),
        expected_away=float((away_points * away).sum()),
    )


def combine_joint_pmfs(first_half: JointScorePMF, second_half: JointScorePMF) -> JointScorePMF:
    home_1h = first_half.pmf.sum(axis=1)
    away_1h = first_half.pmf.sum(axis=0)
    home_2h = second_half.pmf.sum(axis=1)
    away_2h = second_half.pmf.sum(axis=0)

    home_game = np.convolve(home_1h, home_2h)
    away_game = np.convolve(away_1h, away_2h)

    home_points = np.arange(len(home_game), dtype=float)
    away_points = np.arange(len(away_game), dtype=float)
    joint = np.outer(home_game, away_game)
    joint /= joint.sum()

    return JointScorePMF(
        home_points=home_points,
        away_points=away_points,
        pmf=joint,
        expected_home=float((home_points * home_game).sum()),
        expected_away=float((away_points * away_game).sum()),
    )


def build_margin_pmf(joint: JointScorePMF) -> tuple[np.ndarray, np.ndarray]:
    min_margin = int(joint.home_points.min() - joint.away_points.max())
    max_margin = int(joint.home_points.max() - joint.away_points.min())
    margins = np.arange(min_margin, max_margin + 1, dtype=int)
    probs = np.zeros(len(margins), dtype=float)

    for i, h in enumerate(joint.home_points.astype(int)):
        for j, a in enumerate(joint.away_points.astype(int)):
            probs[h - a - min_margin] += joint.pmf[i, j]

    s = probs.sum()
    if s > 0:
        probs /= s
    return margins.astype(float), probs


def score_pair_logprob(joint: JointScorePMF, home_points: int, away_points: int, eps: float = 1e-15) -> float:
    h = int(home_points)
    a = int(away_points)
    if h < 0 or a < 0:
        return np.nan
    if h >= len(joint.home_points) or a >= len(joint.away_points):
        return float(np.log(eps))
    return float(np.log(max(joint.pmf[h, a], eps)))


def margin_logprob(joint: JointScorePMF, margin_value: int, eps: float = 1e-15) -> float:
    margins, probs = build_margin_pmf(joint)
    idx = np.where(margins.astype(int) == int(margin_value))[0]
    if len(idx) == 0:
        return float(np.log(eps))
    return float(np.log(max(probs[idx[0]], eps)))


def prob_home_cover(joint: JointScorePMF, home_spread: float) -> float:
    threshold = -float(home_spread)
    margins, probs = build_margin_pmf(joint)
    return float(probs[margins > threshold].sum())


def prob_over(joint: JointScorePMF, total_line: float) -> float:
    home = joint.home_points[:, None]
    away = joint.away_points[None, :]
    total = home + away
    return float(joint.pmf[total > float(total_line)].sum())


def brier(y_true: pd.Series, p_pred: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(p_pred, errors="coerce")
    mask = y.notna() & p.notna()
    if mask.sum() == 0:
        return float("nan")
    return float(((y[mask] - p[mask]) ** 2).mean())


def offense_long_table(train: pd.DataFrame) -> pd.DataFrame:
    home = train[["home_team", "home_points_1h", "home_points_game"]].rename(
        columns={"home_team": "team", "home_points_1h": "points_1h", "home_points_game": "points_game"}
    )
    away = train[["away_team", "away_points_1h", "away_points_game"]].rename(
        columns={"away_team": "team", "away_points_1h": "points_1h", "away_points_game": "points_game"}
    )
    out = pd.concat([home, away], ignore_index=True)
    out["points_1h"] = pd.to_numeric(out["points_1h"], errors="coerce")
    out["points_game"] = pd.to_numeric(out["points_game"], errors="coerce")
    out = out.loc[out["points_game"].notna() & (out["points_game"] >= 0)].copy()
    return out


def shrunken_first_half_share(train: pd.DataFrame, team: str, prior_points: float, share_min: float, share_max: float) -> float:
    off = offense_long_table(train)
    global_points_game = float(off["points_game"].sum())
    global_points_1h = float(off["points_1h"].sum())
    global_share = global_points_1h / global_points_game if global_points_game > 0 else 0.5

    team_df = off.loc[off["team"].astype(str) == str(team)].copy()
    team_points_game = float(team_df["points_game"].sum()) if len(team_df) else 0.0
    team_points_1h = float(team_df["points_1h"].sum()) if len(team_df) else 0.0

    share = (team_points_1h + prior_points * global_share) / (team_points_game + prior_points)
    return float(np.clip(share, share_min, share_max))


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

    score_summary = pd.DataFrame(rows)

    market_rows = []
    if {"market_spread_game", "actual_home_cover_game", "home_cover_prob_game"} <= set(eval_df.columns):
        mask = eval_df["market_spread_game"].notna() & eval_df["actual_home_cover_game"].notna() & eval_df["home_cover_prob_game"].notna()
        if mask.sum() > 0:
            market_rows.append({"segment": "game", "n_rows": int(mask.sum()), "spread_brier": brier(eval_df.loc[mask, "actual_home_cover_game"], eval_df.loc[mask, "home_cover_prob_game"])})
    if {"market_total_game", "actual_over_game", "over_prob_game"} <= set(eval_df.columns):
        mask = eval_df["market_total_game"].notna() & eval_df["actual_over_game"].notna() & eval_df["over_prob_game"].notna()
        if mask.sum() > 0:
            market_rows.append({"segment": "game", "n_rows": int(mask.sum()), "total_brier": brier(eval_df.loc[mask, "actual_over_game"], eval_df.loc[mask, "over_prob_game"])})

    if {"market_spread_1h", "actual_home_cover_1h", "home_cover_prob_1h"} <= set(eval_df.columns):
        mask = eval_df["market_spread_1h"].notna() & eval_df["actual_home_cover_1h"].notna() & eval_df["home_cover_prob_1h"].notna()
        if mask.sum() > 0:
            market_rows.append({"segment": "1h", "n_rows": int(mask.sum()), "spread_brier": brier(eval_df.loc[mask, "actual_home_cover_1h"], eval_df.loc[mask, "home_cover_prob_1h"])})
    if {"market_total_1h", "actual_over_1h", "over_prob_1h"} <= set(eval_df.columns):
        mask = eval_df["market_total_1h"].notna() & eval_df["actual_over_1h"].notna() & eval_df["over_prob_1h"].notna()
        if mask.sum() > 0:
            market_rows.append({"segment": "1h", "n_rows": int(mask.sum()), "total_brier": brier(eval_df.loc[mask, "actual_over_1h"], eval_df.loc[mask, "over_prob_1h"])})

    if {"market_spread_2h", "actual_home_cover_2h", "home_cover_prob_2h_reg"} <= set(eval_df.columns):
        mask = eval_df["market_spread_2h"].notna() & eval_df["actual_home_cover_2h"].notna() & eval_df["home_cover_prob_2h_reg"].notna()
        if mask.sum() > 0:
            market_rows.append({"segment": "2h", "n_rows": int(mask.sum()), "spread_brier": brier(eval_df.loc[mask, "actual_home_cover_2h"], eval_df.loc[mask, "home_cover_prob_2h_reg"])})

    if {"market_total_2h", "actual_over_2h", "over_prob_2h_reg"} <= set(eval_df.columns):
        mask = eval_df["market_total_2h"].notna() & eval_df["actual_over_2h"].notna() & eval_df["over_prob_2h_reg"].notna()
        if mask.sum() > 0:
            market_rows.append({"segment": "2h", "n_rows": int(mask.sum()), "total_brier": brier(eval_df.loc[mask, "actual_over_2h"], eval_df.loc[mask, "over_prob_2h_reg"])})

    market_summary = pd.DataFrame(market_rows)
    if market_summary.empty:
        return score_summary

    out = score_summary.copy()
    for metric in ["spread_brier", "total_brier"]:
        temp = market_summary[["segment", "n_rows", metric]].dropna()
        if not temp.empty:
            out = out.merge(temp, on=["segment", "n_rows"], how="left")
    return out


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

        pi_train = train.rename(columns={"home_points_game": "home_score", "away_points_game": "away_score"}).copy()
        pi_result = run_pi_benchmark_for_game(
            games_df=pi_train,
            home_team=home_team,
            away_team=away_team,
            neutral_site=neutral_site,
            rho_offense=0.90,
            rho_defense=0.90,
            k_offense=0.10,
            k_defense=0.10,
        )

        game_mu_home = float(pi_result.expected_home_score)
        game_mu_away = float(pi_result.expected_away_score)

        home_share_1h = shrunken_first_half_share(train, home_team, args.share_prior_points, args.share_min, args.share_max)
        away_share_1h = shrunken_first_half_share(train, away_team, args.share_prior_points, args.share_min, args.share_max)

        mu_home_1h = max(game_mu_home * home_share_1h, 0.01)
        mu_away_1h = max(game_mu_away * away_share_1h, 0.01)
        mu_home_2h = max(game_mu_home - mu_home_1h, 0.01)
        mu_away_2h = max(game_mu_away - mu_away_1h, 0.01)

        kappa_home_1h = estimate_kappa(train["home_points_1h"])
        kappa_away_1h = estimate_kappa(train["away_points_1h"])
        kappa_home_2h = estimate_kappa(train["home_points_2h_reg"])
        kappa_away_2h = estimate_kappa(train["away_points_2h_reg"])

        pred_1h = build_joint_pmf(mu_home_1h, mu_away_1h, kappa_home_1h, kappa_away_1h, args.support_max)
        pred_2h = build_joint_pmf(mu_home_2h, mu_away_2h, kappa_home_2h, kappa_away_2h, args.support_max)
        pred_game = combine_joint_pmfs(pred_1h, pred_2h)

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

            "score_logprob_1h": score_pair_logprob(pred_1h, actual_home_1h, actual_away_1h),
            "score_logprob_2h": score_pair_logprob(pred_2h, actual_home_2h, actual_away_2h),
            "score_logprob_game": score_pair_logprob(pred_game, actual_home_game, actual_away_game),

            "margin_logprob_1h": margin_logprob(pred_1h, actual_margin_1h),
            "margin_logprob_2h": margin_logprob(pred_2h, actual_margin_2h),
            "margin_logprob_game": margin_logprob(pred_game, actual_margin_game),

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

            "home_share_1h": home_share_1h,
            "away_share_1h": away_share_1h,
        }

        row["home_cover_prob_game"] = prob_home_cover(pred_game, float(test["market_spread_game"])) if pd.notna(test.get("market_spread_game")) else np.nan
        row["over_prob_game"] = prob_over(pred_game, float(test["market_total_game"])) if pd.notna(test.get("market_total_game")) else np.nan

        row["home_cover_prob_1h"] = prob_home_cover(pred_1h, float(test["market_spread_1h"])) if pd.notna(test.get("market_spread_1h")) else np.nan
        row["over_prob_1h"] = prob_over(pred_1h, float(test["market_total_1h"])) if pd.notna(test.get("market_total_1h")) else np.nan

        row["home_cover_prob_2h_reg"] = prob_home_cover(pred_2h, float(test["market_spread_2h"])) if pd.notna(test.get("market_spread_2h")) else np.nan
        row["over_prob_2h_reg"] = prob_over(pred_2h, float(test["market_total_2h"])) if pd.notna(test.get("market_total_2h")) else np.nan

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
        print("\nSummary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
