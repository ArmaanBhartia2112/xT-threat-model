from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ml.config import PipelineConfig


FEATURE_COLUMNS = [
    "start_zone",
    "end_zone",
    "start_goal_distance",
    "end_goal_distance",
    "start_goal_direction_angle",
    "end_goal_direction_angle",
    "start_goal_mouth_angle",
    "end_goal_mouth_angle",
    "under_pressure",
    "pressure_score",
    "game_state_code",
    "game_phase_code",
    "progressive_flag",
    "action_type_code",
    "action_distance",
]


@dataclass
class TrainedModel:
    model: xgb.XGBClassifier
    feature_columns: list[str]
    validation_auc: float


def build_shot_lookahead_target(events: pd.DataFrame, actions: pd.DataFrame, lookahead: int) -> pd.Series:
    ev = events[["match_id", "index", "team", "possession", "type"]].copy()
    ev = ev.sort_values(["match_id", "index"]).reset_index(drop=True)

    labels = []
    for _, act in actions.iterrows():
        subset = ev[
            (ev["match_id"] == act["match_id"])
            & (ev["team"] == act["team"])
            & (ev["possession"] == act["possession"])
            & (ev["index"] > act["index"])
        ].head(lookahead)
        labels.append(int((subset["type"] == "Shot").any()))
    return pd.Series(labels, index=actions.index, dtype=int)


def train_xgboost(actions: pd.DataFrame, cfg: PipelineConfig) -> TrainedModel:
    X = actions[FEATURE_COLUMNS].fillna(0.0)
    y = actions["target_shot_next_5"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=cfg.random_state,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    auc = float(roc_auc_score(y_val, val_pred)) if len(np.unique(y_val)) > 1 else 0.5

    return TrainedModel(model=model, feature_columns=FEATURE_COLUMNS, validation_auc=auc)


def append_start_end_shot_probs(actions: pd.DataFrame, trained: TrainedModel) -> pd.DataFrame:
    out = actions.copy()

    base = out[trained.feature_columns].fillna(0.0).copy()
    end_prob = trained.model.predict_proba(base)[:, 1]

    start_state = base.copy()
    start_state["end_zone"] = start_state["start_zone"]
    start_state["end_goal_distance"] = start_state["start_goal_distance"]
    start_state["end_goal_direction_angle"] = start_state["start_goal_direction_angle"]
    start_state["end_goal_mouth_angle"] = start_state["start_goal_mouth_angle"]
    start_prob = trained.model.predict_proba(start_state)[:, 1]

    out["shot_prob_start"] = start_prob
    out["shot_prob_end"] = end_prob
    return out
