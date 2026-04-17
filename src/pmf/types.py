from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScorePMF:
    home_points: np.ndarray
    away_points: np.ndarray
    pmf: np.ndarray
    segment_name: str

    def total_probability(self) -> float:
        return float(self.pmf.sum())


@dataclass
class MarginPMF:
    margins: np.ndarray
    pmf: np.ndarray
    segment_name: str

    def total_probability(self) -> float:
        return float(self.pmf.sum())

    def mean_margin(self) -> float:
        return float((self.margins * self.pmf).sum())


@dataclass
class SegmentScoreTargets:
    home_points_1h: np.ndarray
    away_points_1h: np.ndarray
    home_points_2h: np.ndarray
    away_points_2h: np.ndarray
    home_points_game: np.ndarray
    away_points_game: np.ndarray


@dataclass
class MarginTargets:
    margin_1h: np.ndarray
    margin_2h: np.ndarray
    margin_game: np.ndarray
