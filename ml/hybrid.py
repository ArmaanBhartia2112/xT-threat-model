from __future__ import annotations

import numpy as np
import pandas as pd


def compute_hybrid_xt(actions: pd.DataFrame, xt_surface: np.ndarray, alpha: float = 0.5) -> pd.DataFrame:
    out = actions.copy()
    out["xt_zone_start"] = out["start_zone"].astype(int).map(lambda z: float(xt_surface[z]))
    out["xt_zone_end"] = out["end_zone"].astype(int).map(lambda z: float(xt_surface[z]))
    out["xt_zone_delta"] = out["xt_zone_end"] - out["xt_zone_start"]

    out["xt_ml_delta"] = out["shot_prob_end"] - out["shot_prob_start"]
    out["xt_value"] = alpha * out["xt_zone_delta"] + (1 - alpha) * out["xt_ml_delta"]
    return out
