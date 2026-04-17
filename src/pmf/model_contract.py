from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .types import MarginPMF, ScorePMF


@dataclass
class PregamePMFBundle:
    score_pmf_1h: ScorePMF
    score_pmf_2h: ScorePMF
    score_pmf_game: ScorePMF
    margin_pmf_1h: MarginPMF
    margin_pmf_2h: MarginPMF
    margin_pmf_game: MarginPMF


class PregameMarginPMFModel(Protocol):
    def fit(self, games_df: pd.DataFrame) -> "PregameMarginPMFModel":
        ...

    def predict_game(self, schedule_row: pd.Series) -> PregamePMFBundle:
        ...
