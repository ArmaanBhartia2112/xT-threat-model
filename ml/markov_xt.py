from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import PipelineConfig


def compute_zone_probabilities(
    actions: pd.DataFrame, shots: pd.DataFrame, cfg: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = cfg.grid_x * cfg.grid_y

    shot_counts = np.zeros(n)
    move_counts = np.zeros(n)
    total_counts = np.zeros(n)
    goal_shot_counts = np.zeros(n)

    trans_counts = np.zeros((n, n))

    for _, row in actions.iterrows():
        s = int(row["start_zone"])
        e = int(row["end_zone"])
        if s < 0 or s >= n or e < 0 or e >= n:
            continue
        total_counts[s] += 1
        move_counts[s] += 1
        trans_counts[s, e] += 1

    for _, row in shots.iterrows():
        s = int(row["start_zone"])
        if s < 0 or s >= n:
            continue
        total_counts[s] += 1
        shot_counts[s] += 1
        goal_shot_counts[s] += int(row["is_goal"])

    shot_prob = np.divide(shot_counts, total_counts, out=np.zeros_like(shot_counts), where=total_counts > 0)
    move_prob = np.divide(move_counts, total_counts, out=np.zeros_like(move_counts), where=total_counts > 0)
    goal_prob = np.divide(
        goal_shot_counts,
        shot_counts,
        out=np.zeros_like(goal_shot_counts),
        where=shot_counts > 0,
    )

    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans = np.divide(trans_counts, row_sums, out=np.zeros_like(trans_counts), where=row_sums > 0)

    return shot_prob, move_prob, goal_prob, trans


def value_iteration(
    shot_prob: np.ndarray,
    move_prob: np.ndarray,
    goal_prob: np.ndarray,
    transition: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    xt = np.zeros_like(shot_prob, dtype=float)

    for _ in range(cfg.xt_iterations):
        xt = shot_prob * goal_prob + move_prob * (transition @ xt)

    return xt
