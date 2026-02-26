from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.aggregate import player_aggregation, team_aggregation, validate_correlations
from ml.config import PipelineConfig
from ml.data_ingestion import load_statsbomb_events
from ml.features import add_spatial_features, build_game_state, encode_context_features
from ml.hybrid import compute_hybrid_xt
from ml.markov_xt import compute_zone_probabilities, value_iteration
from ml.model import append_start_end_shot_probs, build_shot_lookahead_target, train_xgboost


def run_pipeline(cfg: PipelineConfig | None = None) -> dict[str, float]:
    cfg = cfg or PipelineConfig()

    cfg.data_raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_statsbomb_events(cfg)

    actions = pd.concat([loaded.passes, loaded.carries], ignore_index=True)
    actions = add_spatial_features(actions, cfg)
    shots = add_spatial_features(loaded.shots, cfg)

    actions = build_game_state(actions, shots)
    actions = encode_context_features(actions)

    actions["target_shot_next_5"] = build_shot_lookahead_target(
        loaded.events,
        actions,
        lookahead=cfg.shot_lookahead_actions,
    )

    shot_prob, move_prob, goal_prob, transition = compute_zone_probabilities(actions, shots, cfg)
    xt_surface = value_iteration(shot_prob, move_prob, goal_prob, transition, cfg)

    trained = train_xgboost(actions, cfg)
    actions = append_start_end_shot_probs(actions, trained)

    shot_meta = shots[["match_id", "team", "player", "minute", "second", "shot_statsbomb_xg", "is_goal"]].copy()
    shot_meta = shot_meta.rename(columns={"shot_statsbomb_xg": "xg"})

    actions = actions.merge(
        shot_meta,
        on=["match_id", "team", "player", "minute", "second"],
        how="left",
    )
    actions["action_xg"] = actions["xg"].fillna(0.0)
    actions["is_action_goal"] = actions["is_goal"].fillna(0).astype(int)

    actions = compute_hybrid_xt(actions, xt_surface, alpha=0.5)

    player_stats = player_aggregation(actions)
    team_stats = team_aggregation(actions)
    corrs = validate_correlations(player_stats)

    actions_path = cfg.data_processed_dir / "actions_hybrid_xt.parquet"
    players_path = cfg.data_processed_dir / "player_stats.parquet"
    teams_path = cfg.data_processed_dir / "team_stats.parquet"
    actions.to_parquet(actions_path, index=False)
    player_stats.to_parquet(players_path, index=False)
    team_stats.to_parquet(teams_path, index=False)

    np.save(cfg.artifacts_dir / "xt_surface.npy", xt_surface)
    np.save(cfg.artifacts_dir / "transition_matrix.npy", transition)
    joblib.dump(trained.model, cfg.artifacts_dir / "xgboost_shot_model.joblib")

    metadata = {
        "competition_id": cfg.competition_id,
        "season_id": cfg.season_id,
        "grid": [cfg.grid_y, cfg.grid_x],
        "validation_auc": trained.validation_auc,
        **corrs,
    }
    (cfg.artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
