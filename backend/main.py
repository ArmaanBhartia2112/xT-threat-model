from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

app = FastAPI(title="Hybrid xT API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Store:
    xt_surface: np.ndarray | None = None
    model = None
    actions: pd.DataFrame | None = None
    players: pd.DataFrame | None = None


store = Store()


@app.on_event("startup")
def startup() -> None:
    surface_path = ARTIFACTS_DIR / "xt_surface.npy"
    model_path = ARTIFACTS_DIR / "xgboost_shot_model.joblib"
    actions_path = PROCESSED_DIR / "actions_hybrid_xt.parquet"
    players_path = PROCESSED_DIR / "player_stats.parquet"

    if not (surface_path.exists() and model_path.exists() and actions_path.exists() and players_path.exists()):
        return

    store.xt_surface = np.load(surface_path)
    store.model = joblib.load(model_path)
    store.actions = pd.read_parquet(actions_path)
    store.players = pd.read_parquet(players_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/surface")
def surface() -> dict[str, list[list[float]]]:
    if store.xt_surface is None:
        raise HTTPException(status_code=404, detail="xT surface not found. Run pipeline first.")

    arr = store.xt_surface.reshape(12, 16).tolist()
    return {"grid": arr}


@app.get("/players")
def players() -> list[dict]:
    if store.players is None:
        raise HTTPException(status_code=404, detail="Player table not found. Run pipeline first.")
    return store.players.to_dict(orient="records")


@app.get("/player-actions")
def player_actions(player_name: str = Query(..., min_length=2)) -> list[dict]:
    if store.actions is None:
        raise HTTPException(status_code=404, detail="Actions table not found. Run pipeline first.")

    subset = store.actions[store.actions["player"] == player_name].copy()
    if subset.empty:
        return []

    cols = [
        "id",
        "match_id",
        "team",
        "player",
        "minute",
        "type",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "xt_value",
        "progressive_flag",
        "under_pressure",
        "pressure_score",
    ]
    existing = [c for c in cols if c in subset.columns]
    subset = subset[existing].sort_values(["match_id", "minute"]).reset_index(drop=True)
    return subset.to_dict(orient="records")


@app.get("/player-stats")
def player_stats(player_name: str = Query(..., min_length=2)) -> dict:
    if store.players is None:
        raise HTTPException(status_code=404, detail="Player table not found. Run pipeline first.")

    subset = store.players[store.players["player"] == player_name]
    if subset.empty:
        raise HTTPException(status_code=404, detail="Player not found")
    return subset.iloc[0].to_dict()
