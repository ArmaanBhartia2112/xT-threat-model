from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _minutes_played(actions: pd.DataFrame) -> pd.DataFrame:
    mins = (
        actions.groupby(["match_id", "player"], as_index=False)["minute"]
        .max()
        .rename(columns={"minute": "minutes_in_match"})
    )
    total = mins.groupby("player", as_index=False)["minutes_in_match"].sum()
    total["minutes_in_match"] = total["minutes_in_match"].clip(lower=1)
    return total


def player_aggregation(actions: pd.DataFrame) -> pd.DataFrame:
    mins = _minutes_played(actions)

    agg = (
        actions.groupby("player", as_index=False)
        .agg(
            total_xt=("xt_value", "sum"),
            total_actions=("id", "count"),
            progressive_actions=("progressive_flag", "sum"),
            pressure_action_rate=("under_pressure", "mean"),
            goals=("is_action_goal", "sum"),
            xg=("action_xg", "sum"),
        )
        .merge(mins, on="player", how="left")
    )

    agg["xt_per_90"] = agg["total_xt"] * 90.0 / agg["minutes_in_match"]
    agg["goals_per_90"] = agg["goals"] * 90.0 / agg["minutes_in_match"]
    agg["xg_per_90"] = agg["xg"] * 90.0 / agg["minutes_in_match"]
    agg = agg.sort_values("xt_per_90", ascending=False).reset_index(drop=True)

    return agg


def team_aggregation(actions: pd.DataFrame) -> pd.DataFrame:
    agg = (
        actions.groupby("team", as_index=False)
        .agg(
            total_xt=("xt_value", "sum"),
            total_actions=("id", "count"),
            progressive_actions=("progressive_flag", "sum"),
            pressure_action_rate=("under_pressure", "mean"),
        )
        .sort_values("total_xt", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def validate_correlations(player_stats: pd.DataFrame) -> dict[str, float]:
    valid = player_stats.dropna(subset=["xt_per_90", "goals_per_90", "xg_per_90"])
    if len(valid) < 3:
        return {
            "pearson_xt_goals": np.nan,
            "spearman_xt_goals": np.nan,
            "pearson_xt_xg": np.nan,
            "spearman_xt_xg": np.nan,
        }

    p_g = pearsonr(valid["xt_per_90"], valid["goals_per_90"]).statistic
    s_g = spearmanr(valid["xt_per_90"], valid["goals_per_90"]).statistic
    p_xg = pearsonr(valid["xt_per_90"], valid["xg_per_90"]).statistic
    s_xg = spearmanr(valid["xt_per_90"], valid["xg_per_90"]).statistic

    return {
        "pearson_xt_goals": float(p_g),
        "spearman_xt_goals": float(s_g),
        "pearson_xt_xg": float(p_xg),
        "spearman_xt_xg": float(s_xg),
    }
