from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import PipelineConfig


def assign_zone(x: pd.Series, y: pd.Series, cfg: PipelineConfig) -> tuple[pd.Series, pd.Series, pd.Series]:
    zx = np.clip((x / cfg.pitch_length * cfg.grid_x).astype(int), 0, cfg.grid_x - 1)
    zy = np.clip((y / cfg.pitch_width * cfg.grid_y).astype(int), 0, cfg.grid_y - 1)
    z = zy * cfg.grid_x + zx
    return zx, zy, z


def _goal_mouth_angle(x: pd.Series, y: pd.Series, cfg: PipelineConfig) -> pd.Series:
    left_dx = cfg.goal_center_x - x
    left_dy = cfg.goal_left_y - y
    right_dx = cfg.goal_center_x - x
    right_dy = cfg.goal_right_y - y
    num = np.abs(left_dx * right_dy - left_dy * right_dx)
    den = left_dx * right_dx + left_dy * right_dy
    return np.arctan2(num, den)


def add_spatial_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()

    out["start_zone_x"], out["start_zone_y"], out["start_zone"] = assign_zone(
        out["start_x"], out["start_y"], cfg
    )
    out["end_zone_x"], out["end_zone_y"], out["end_zone"] = assign_zone(out["end_x"], out["end_y"], cfg)

    out["start_goal_distance"] = np.hypot(cfg.goal_center_x - out["start_x"], cfg.goal_center_y - out["start_y"])
    out["end_goal_distance"] = np.hypot(cfg.goal_center_x - out["end_x"], cfg.goal_center_y - out["end_y"])

    out["start_goal_direction_angle"] = np.arctan2(cfg.goal_center_y - out["start_y"], cfg.goal_center_x - out["start_x"])
    out["end_goal_direction_angle"] = np.arctan2(cfg.goal_center_y - out["end_y"], cfg.goal_center_x - out["end_x"])

    out["start_goal_mouth_angle"] = _goal_mouth_angle(out["start_x"], out["start_y"], cfg)
    out["end_goal_mouth_angle"] = _goal_mouth_angle(out["end_x"], out["end_y"], cfg)

    out["progressive_flag"] = (
        out["end_goal_distance"] <= (2.0 / 3.0) * out["start_goal_distance"]
    ).astype(int)
    out["action_distance"] = np.hypot(out["end_x"] - out["start_x"], out["end_y"] - out["start_y"])

    return out


def build_game_state(actions: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    out = actions.copy()
    goals = shots[shots["is_goal"] == 1][["match_id", "period", "minute", "second", "team"]].copy()
    goals["goal_for"] = 1

    out = out.sort_values(["match_id", "period", "minute", "second", "index"]).reset_index(drop=True)

    out["home_goals"] = 0
    out["away_goals"] = 0

    for match_id, group_idx in out.groupby("match_id").groups.items():
        idx = list(group_idx)
        match_actions = out.loc[idx].copy()
        teams = match_actions["team"].dropna().unique().tolist()
        if len(teams) < 2:
            continue
        home_team, away_team = teams[0], teams[1]

        match_goals = goals[goals["match_id"] == match_id].copy()
        match_goals = match_goals.sort_values(["period", "minute", "second"]) if not match_goals.empty else match_goals

        hg = 0
        ag = 0
        gptr = 0
        gvals = match_goals.to_dict("records")

        for ridx in idx:
            row = out.loc[ridx]
            while gptr < len(gvals):
                g = gvals[gptr]
                if (
                    (g["period"], g["minute"], g["second"])
                    <= (row["period"], row["minute"], row["second"])
                ):
                    if g["team"] == home_team:
                        hg += 1
                    elif g["team"] == away_team:
                        ag += 1
                    gptr += 1
                else:
                    break
            out.at[ridx, "home_goals"] = hg
            out.at[ridx, "away_goals"] = ag
            if row["team"] == home_team:
                diff = hg - ag
            else:
                diff = ag - hg
            out.at[ridx, "score_diff"] = diff

    out["game_state"] = np.where(
        out["score_diff"] > 0,
        "winning",
        np.where(out["score_diff"] < 0, "losing", "drawing"),
    )

    out["game_phase"] = pd.cut(
        out["minute"],
        bins=[-1, 15, 60, 90, 200],
        labels=["early", "mid", "late", "extra"],
    ).astype(str)

    return out


def encode_context_features(actions: pd.DataFrame) -> pd.DataFrame:
    out = actions.copy()
    out["game_state_code"] = out["game_state"].map({"losing": 0, "drawing": 1, "winning": 2}).fillna(1)
    out["game_phase_code"] = out["game_phase"].map({"early": 0, "mid": 1, "late": 2, "extra": 3}).fillna(1)
    out["action_type_code"] = out["type"].map({"Pass": 0, "Carry": 1}).fillna(0)
    out["under_pressure"] = out["under_pressure"].fillna(0).astype(int)
    out["pressure_score"] = out["pressure_score"].fillna(0.0).astype(float)
    return out
