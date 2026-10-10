import warnings
from collections.abc import Mapping

import numpy as np


def compute_crafter_score_from_percent(success_rates_percent: Mapping[str, float]) -> float:
    percents = np.asarray(list(success_rates_percent.values()), dtype=np.float32)
    if percents.size == 0:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.exp(np.nanmean(np.log1p(percents))) - 1.0)


def compute_crafter_score_from_fraction(success_rates_fraction: Mapping[str, float]) -> float:
    return compute_crafter_score_from_percent(
        {key: float(value) * 100.0 for key, value in success_rates_fraction.items()}
    )
