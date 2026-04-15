from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, lgamma, sqrt
from typing import Literal

import numpy as np


@dataclass
class JointPMF:
    home_points: np.ndarray
    away_points: np.ndarray
    pmf: np.ndarray
    expected_home: float
    expected_away: float

    @property
    def expected_total(self) -> float:
        return self.expected_home + self.expected_away

    @property
    def expected_margin_home(self) -> float:
        return self.expected_home - self.expected_away


def _validate_mu_kappa(mu: float, kappa: float, label: str) -> None:
    if mu < 0:
        raise ValueError(f"{label} mean must be nonnegative")
    if kappa <= 0:
        raise ValueError(f"{label} dispersion must be positive")


def _nb_pmf_grid(mu: float, kappa: float, max_points: int) -> np.ndarray:
    """
    Negative binomial PMF using mean/dispersion parameterization:
        Var(X) = mu + mu^2 / kappa
    """
    _validate_mu_kappa(mu, kappa, "negative binomial")
    if max_points < 0:
        raise ValueError("max_points must be nonnegative")

    if mu == 0:
        out = np.zeros(max_points + 1, dtype=float)
        out[0] = 1.0
        return out

    r = float(kappa)
    p = r / (r + mu)  # success probability

    probs = np.zeros(max_points + 1, dtype=float)
    for x in range(max_points + 1):
        log_pmf = (
            lgamma(x + r)
            - lgamma(r)
            - lgamma(x + 1)
            + r * np.log(p)
            + x * np.log(1.0 - p)
        )
        probs[x] = exp(log_pmf)

    total = probs.sum()
    if total <= 0:
        raise ValueError("PMF normalization failed")
    probs /= total
    return probs


def _default_max_points(mu: float, kappa: float, floor_points: int = 60) -> int:
    variance = mu + (mu * mu) / kappa
    std = sqrt(max(variance, 1e-9))
    return max(floor_points, int(ceil(mu + 6.0 * std)))


def build_independent_joint_pmf(
    mu_home: float,
    mu_away: float,
    kappa_home: float = 18.0,
    kappa_away: float = 18.0,
    max_home_points: int | None = None,
    max_away_points: int | None = None,
) -> JointPMF:
    """
    Benchmark-only joint PMF:
    - home score ~ NegBin(mu_home, kappa_home)
    - away score ~ NegBin(mu_away, kappa_away)
    - assumes independence

    This is a baseline PMF layer for benchmark pricing.
    The full possession-based production model will replace this.
    """
    _validate_mu_kappa(mu_home, kappa_home, "home")
    _validate_mu_kappa(mu_away, kappa_away, "away")

    if max_home_points is None:
        max_home_points = _default_max_points(mu_home, kappa_home)
    if max_away_points is None:
        max_away_points = _default_max_points(mu_away, kappa_away)

    home_points = np.arange(max_home_points + 1, dtype=int)
    away_points = np.arange(max_away_points + 1, dtype=int)

    pmf_home = _nb_pmf_grid(mu_home, kappa_home, max_home_points)
    pmf_away = _nb_pmf_grid(mu_away, kappa_away, max_away_points)

    joint = np.outer(pmf_home, pmf_away)
    joint /= joint.sum()

    expected_home = float((home_points * pmf_home).sum())
    expected_away = float((away_points * pmf_away).sum())

    return JointPMF(
        home_points=home_points,
        away_points=away_points,
        pmf=joint,
        expected_home=expected_home,
        expected_away=expected_away,
    )


def home_win_prob(joint: JointPMF) -> float:
    hp = joint.home_points[:, None]
    ap = joint.away_points[None, :]
    return float(joint.pmf[hp > ap].sum())


def away_win_prob(joint: JointPMF) -> float:
    hp = joint.home_points[:, None]
    ap = joint.away_points[None, :]
    return float(joint.pmf[hp < ap].sum())


def tie_prob(joint: JointPMF) -> float:
    hp = joint.home_points[:, None]
    ap = joint.away_points[None, :]
    return float(joint.pmf[hp == ap].sum())


def home_margin_greater_than(joint: JointPMF, threshold: float) -> float:
    hp = joint.home_points[:, None]
    ap = joint.away_points[None, :]
    return float(joint.pmf[(hp - ap) > threshold].sum())


def away_margin_greater_than(joint: JointPMF, threshold: float) -> float:
    hp = joint.home_points[:, None]
    ap = joint.away_points[None, :]
    return float(joint.pmf[(ap - hp) > threshold].sum())


def total_over_prob(joint: JointPMF, total: float) -> float:
    hp = joint.home_points[:, None]
    ap = joint.away_points[None, :]
    return float(joint.pmf[(hp + ap) > total].sum())


def total_under_prob(joint: JointPMF, total: float) -> float:
    hp = joint.home_points[:, None]
    ap = joint.away_points[None, :]
    return float(joint.pmf[(hp + ap) < total].sum())


def team_total_over_prob(
    joint: JointPMF,
    team: Literal["home", "away"],
    total: float,
) -> float:
    if team == "home":
        pts = joint.home_points
        marginal = joint.pmf.sum(axis=1)
    elif team == "away":
        pts = joint.away_points
        marginal = joint.pmf.sum(axis=0)
    else:
        raise ValueError("team must be 'home' or 'away'")

    return float(marginal[pts > total].sum())


def team_total_under_prob(
    joint: JointPMF,
    team: Literal["home", "away"],
    total: float,
) -> float:
    if team == "home":
        pts = joint.home_points
        marginal = joint.pmf.sum(axis=1)
    elif team == "away":
        pts = joint.away_points
        marginal = joint.pmf.sum(axis=0)
    else:
        raise ValueError("team must be 'home' or 'away'")

    return float(marginal[pts < total].sum())


def fair_decimal_odds(prob: float) -> float:
    if prob <= 0 or prob >= 1:
        raise ValueError("prob must be strictly between 0 and 1")
    return 1.0 / prob


def fair_american_odds(prob: float) -> int:
    if prob <= 0 or prob >= 1:
        raise ValueError("prob must be strictly between 0 and 1")

    if prob >= 0.5:
        return int(round(-100.0 * prob / (1.0 - prob)))
    return int(round(100.0 * (1.0 - prob) / prob))


__all__ = [
    "JointPMF",
    "build_independent_joint_pmf",
    "home_win_prob",
    "away_win_prob",
    "tie_prob",
    "home_margin_greater_than",
    "away_margin_greater_than",
    "total_over_prob",
    "total_under_prob",
    "team_total_over_prob",
    "team_total_under_prob",
    "fair_decimal_odds",
    "fair_american_odds",
]
