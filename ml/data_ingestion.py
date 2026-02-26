from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsbombpy import sb

from ml.config import PipelineConfig


@dataclass
class LoadedData:
    events: pd.DataFrame
    passes: pd.DataFrame
    carries: pd.DataFrame
    shots: pd.DataFrame


def _safe_xy(val: Any) -> tuple[float, float]:
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return float(val[0]), float(val[1])
    return np.nan, np.nan


def _extract_pressure_score(row: pd.Series) -> float:
    freeze = row.get("freeze_frame")
    start_x = row.get("start_x")
    start_y = row.get("start_y")
    if not isinstance(freeze, list) or np.isnan(start_x) or np.isnan(start_y):
        return 0.0

    score = 0.0
    for actor in freeze:
        if not isinstance(actor, dict):
            continue
        if actor.get("teammate") is True:
            continue
        loc = actor.get("location")
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        dx = float(loc[0]) - float(start_x)
        dy = float(loc[1]) - float(start_y)
        d = float(np.hypot(dx, dy))
        if 0 < d <= 5:
            score += 1.0 / d
    return score


def _parse_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "location" not in out.columns:
        out["location"] = None
    if "end_location" not in out.columns:
        out["end_location"] = None
    out["start_x"], out["start_y"] = zip(*out["location"].map(_safe_xy))
    out["end_x"], out["end_y"] = zip(*out["end_location"].map(_safe_xy))
    out["under_pressure"] = out.get("under_pressure", False).fillna(False).astype(int)
    out["pressure_score"] = out.apply(_extract_pressure_score, axis=1)

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_timedelta(out["timestamp"].astype(str), errors="coerce")
    else:
        out["timestamp"] = pd.to_timedelta("0s")

    out["minute"] = out.get("minute", 0).fillna(0).astype(int)
    out["second"] = out.get("second", 0).fillna(0).astype(int)

    needed = [
        "id",
        "match_id",
        "index",
        "period",
        "timestamp",
        "minute",
        "second",
        "team",
        "player",
        "type",
        "possession",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "under_pressure",
        "pressure_score",
    ]
    for col in needed:
        if col not in out.columns:
            out[col] = np.nan

    return out


def load_statsbomb_events(cfg: PipelineConfig) -> LoadedData:
    matches = sb.matches(competition_id=cfg.competition_id, season_id=cfg.season_id)
    match_ids = matches["match_id"].tolist()

    all_events: list[pd.DataFrame] = []
    for match_id in match_ids:
        events = sb.events(match_id=match_id)
        events["match_id"] = match_id
        all_events.append(events)

    events_df = pd.concat(all_events, ignore_index=True)
    events_df = events_df.sort_values(["match_id", "period", "minute", "second", "index"]).reset_index(
        drop=True
    )

    passes = events_df[events_df["type"] == "Pass"].copy()
    passes["end_location"] = passes.get("pass_end_location")
    pass_outcome = (
        passes["pass_outcome"] if "pass_outcome" in passes.columns else pd.Series(index=passes.index, dtype=object)
    )
    passes = passes[pass_outcome.isna()].copy()

    carries = events_df[events_df["type"] == "Carry"].copy()
    carries["end_location"] = carries.get("carry_end_location")
    carries = carries[carries["carry_end_location"].notna()].copy()

    shots = events_df[events_df["type"] == "Shot"].copy()
    shots["end_location"] = shots.get("shot_end_location")
    shots["is_goal"] = (shots.get("shot_outcome") == "Goal").astype(int)

    passes = _parse_base_columns(passes)
    carries = _parse_base_columns(carries)
    shots = _parse_base_columns(shots)

    return LoadedData(events=events_df, passes=passes, carries=carries, shots=shots)
